#!/bin/bash

echo "Starting application..."
echo "PYTHONPATH is: $PYTHONPATH"
echo "Available GPUs (according to nvidia-smi):"
nvidia-smi || echo "nvidia-smi not found or no GPU detected by it."

echo "PyTorch CUDA available: "
python3 -c "import torch; print(torch.cuda.is_available()); print(f'Device count: {torch.cuda.device_count()}')"
python3 /app/grove/utils/data_loader.py

echo "Application startup script finished. Keeping container alive for now."
sleep infinity