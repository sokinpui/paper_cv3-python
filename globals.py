import torch

try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Web UI running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
except Exception as e:
    print(f"Warning: {e}")
    DEVICE = torch.device("cpu")
