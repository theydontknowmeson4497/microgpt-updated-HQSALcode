import random
import numpy as np
import torch
import time
import os

def get_device() -> torch.device:
    """
    Returns the best available compute device.
    Prefers CUDA (RTX GPU) when available, otherwise falls back to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        print(f"[Device] Using GPU: {props.name} "
              f"({props.total_memory / 1024**3:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available — running on CPU.")
    return device

def set_seed(seed: int = 42):
    """
    Sets deterministic seeds for standard Python, NumPy, and PyTorch.
    Ensures reproducibility of quantum simulations and classical layers.
    When CUDA is available, benchmark mode is ENABLED for extra speed
    (determinism is relaxed since quantum circuits are stochastic anyway).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Allow cuDNN to pick the fastest convolution/matmul kernel.
        # benchmark=True trades perfect reproducibility for speed —
        # acceptable here because quantum circuits introduce inherent randomness.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for reproducibility in hashing
    os.environ['PYTHONHASHSEED'] = str(seed)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Timer:
    """
    A simple context manager to measure execution time.
    """
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
