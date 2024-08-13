poetry run python main.py --device cpu > cpu.profile
poetry run python main.py --device cuda > gpu.profile

echo "---------------------------------------------"
echo "Average time CPU: "
tail -1 cpu.profile
echo "Average time GPU: "
tail -1 gpu.profile
