import random
import numpy as np
import torch
import time
import os

def set_seed(seed: int = 42):
    """
    Sets deterministic seeds for standard Python, NumPy, and PyTorch.
    Ensures reproducibility of quantum simulations and classical layers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configure PyTorch to be as deterministic as possible
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
