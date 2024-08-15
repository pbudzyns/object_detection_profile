import abc
import argparse
import json
import pathlib
from time import perf_counter
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage, ToTensor
from torchvision.utils import draw_bounding_boxes

with open("label_names.json", "r") as f:
    COCO_INSTANCE_CATEGORY_NAMES = json.load(f)


def profiler_wrapper(f: Callable, name: str) -> Callable:
    # Timing wrapper for third-party methods.
    def wrapped(*args, **kwargs):
        t = perf_counter()
        res = f(*args, **kwargs)
        print(f"[{name}]\tcompleted in", (perf_counter() - t) * 1000, "ms")
        return res
    return wrapped


class DataLoader:
    def __init__(self, image_folder: str):
        self.image_files = pathlib.Path(image_folder).iterdir()
        self.images = self.load_images()

    def load_images(self) -> list[tuple[str, Image]]:
        return [(filename.name, Image.open(filename)) for filename in self.image_files]

    def __iter__(self):
        return iter(self.images)


class Profiler(abc.ABC):

    def __init__(self, profile: bool = True):
        self.profile = profile
        self.records = []

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def average_elapsed(self) -> np.float32:
        return np.mean(self.records)

    def __call__(self, *args, **kwargs):
        if not self.profile:
            return self.forward(*args, **kwargs)

        t = perf_counter()
        res = self.forward(*args, **kwargs)
        elapsed = (perf_counter() - t) * 1000
        print(
            f"[{self.__class__.__name__}.__call__]\tcompleted in",
            elapsed,
            "ms",
        )
        self.records.append(elapsed)
        return res


class PreProcessor(Profiler):

    def __init__(self, target_size: tuple[int, int] = (300, 400), profile: bool = True):
        super().__init__(profile)
        steps = [ToImage(), ToDtype(torch.float32, scale=True), Resize(target_size)]

        if profile:
            for i in range(len(steps)):
                steps[i].forward = profiler_wrapper(
                    steps[i].forward,
                    self.__class__.__name__ + "." + steps[i].__class__.__name__,
                )
            self.__call__ = profiler_wrapper(
                self.__call__, f"{self.__class__.__name__}.__call__"
            )
        self.transform = Compose(steps)

    def forward(self, image_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "original_images": [image_batch],
            "model_inputs": self.transform(image_batch).unsqueeze(0),
        }


class ObjectDetectionModel(Profiler):

    def __init__(
        self, device: str = "cpu", profile: bool = True, host_device: str = "cpu"
    ):
        super().__init__(profile)
        self.device = device
        self.host_device = host_device
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
        if profile:
            for key, value in self.model.named_modules():
                value.forward = profiler_wrapper(
                    value.forward, f"{self.__class__.__name__}.{key}"
                )
        self.model.eval()

    def forward(self, image_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = image_batch["model_inputs"]
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        return image_batch | {"model_outputs": outputs}


class PostProcessor(Profiler):

    def __init__(self, score_threshold: float = 0.85, profile=True):
        super().__init__(profile)
        self.threshold = score_threshold
        if profile:
            self.select_boxes = profiler_wrapper(
                self.select_boxes, f"{self.__class__.__name__}.select_boxes"
            )
            self.draw_boxes = profiler_wrapper(
                self.draw_boxes, f"{self.__class__.__name__}.draw_boxes"
            )
            self.__call__ = profiler_wrapper(
                self.__call__, f"{self.__class__.__name__}.__call__"
            )

    def select_boxes(self, outputs):
        # Select boxes for which score exceeds given threshold.
        for output in outputs:
            mask = output["scores"] >= self.threshold
            for key, value in output.items():
                output[key] = value[mask]
        return outputs

    @classmethod
    def draw_boxes(cls, outputs):
        # Draw bounding boxes on an image.
        detected_images = []
        for original_image, model_input, model_output in zip(
            outputs["original_images"],
            outputs["model_inputs"],
            outputs["model_outputs"],
        ):
            og = ToTensor()(original_image)
            og_c, og_w, og_h = og.shape  # original shapes
            sc_c, sc_w, sc_h = model_input.shape  # scaled shapes
            w_const = og_w / sc_w  # horizontal scaling factor
            h_const = og_h / sc_h  # vertical scaling factor
            # Rescale bounding boxes to put them on the original image.
            model_output["boxes"][:, 0] *= h_const
            model_output["boxes"][:, 1] *= w_const
            model_output["boxes"][:, 2] *= h_const
            model_output["boxes"][:, 3] *= w_const

            detected_images.append(
                draw_bounding_boxes(
                    og,
                    model_output["boxes"],
                    colors="red",
                    width=5,
                    labels=[
                        COCO_INSTANCE_CATEGORY_NAMES[i] for i in model_output["labels"]
                    ],
                )
            )
        return detected_images

    def forward(self, model_outputs):
        model_outputs["model_outputs"] = self.select_boxes(
            model_outputs["model_outputs"]
        )
        return self.draw_boxes(model_outputs)


class Pipeline(Profiler):

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: ObjectDetectionModel,
        post_processor: PostProcessor,
        profile: bool = True,
    ):
        super().__init__(profile)
        self.pre_processor = pre_processor
        self.model = model
        self.post_processor = post_processor

    def forward(self, image):
        inputs = self.pre_processor(image)
        outputs = self.model(inputs)
        return self.post_processor(outputs)


def save_output(image: torch.Tensor, name: str, folder: str) -> None:
    output_folder = pathlib.Path(folder)
    if not output_folder.exists():
        output_folder.mkdir()
    for img in image:
        F.to_pil_image(img).save(output_folder / name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        help="Accelerator device to run model on.",
        default="cpu",
    )
    parser.add_argument(
        "--no-profile",
        help="Disable profiling printing.",
        action="store_false",
        default=True,
    )

    args = parser.parse_args()

    dataloader = DataLoader("./sample_images")

    profile = args.no_profile
    pipeline = Pipeline(
        PreProcessor(profile=profile),
        ObjectDetectionModel(device=args.device, profile=profile),
        PostProcessor(profile=profile),
        profile=profile,
    )

    for name, image in dataloader:
        results = pipeline(image)
        save_output(results, name, "./outputs")

    if profile:
        print("Total Average:", pipeline.average_elapsed(), "ms")
