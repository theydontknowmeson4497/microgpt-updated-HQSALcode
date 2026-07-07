import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import requests

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

class TextLMDataset(Dataset):
    """
    Character-level language modeling dataset.
    Uses UTF-8 text encoded into byte values 0..127.
    """
    def __init__(self, tokens: list[int], seq_len: int, num_samples: int, seed: int = None):
        self.seq_len = seq_len
        total_positions = len(tokens) - seq_len - 1
        if total_positions <= 0:
            raise ValueError("Text is too short for the requested sequence length.")

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        if num_samples is None or num_samples > total_positions:
            self.start_indices = list(range(total_positions))
        else:
            self.start_indices = rng.sample(range(total_positions), num_samples)
            self.start_indices.sort()

        self.tokens = tokens

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        x_tokens = self.tokens[start:start + self.seq_len]
        y_tokens = self.tokens[start + 1:start + 1 + self.seq_len]
        x = torch.tensor(x_tokens, dtype=torch.long)
        y = torch.tensor(y_tokens, dtype=torch.long)
        return x, y


def download_tiny_shakespeare(cache_dir: str) -> str:
    """Download Tiny Shakespeare text to cache_dir if missing."""
    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, "tiny_shakespeare.txt")
    if os.path.exists(target_path):
        return target_path

    response = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
    response.raise_for_status()
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    return target_path


def load_tiny_shakespeare(cache_dir: str, seq_len: int, num_samples: int, val_samples: int, seed: int = 42):
    """Returns train and validation datasets from Tiny Shakespeare."""
    path = download_tiny_shakespeare(cache_dir)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = [ord(ch) if 0 <= ord(ch) < 128 else 0 for ch in text]
    total_positions = len(tokens) - seq_len - 1
    if total_positions <= 0:
        raise ValueError("Text is too short to build a dataset with the requested seq_len.")

    split_point = max(int(total_positions * 0.9), seq_len + 1)
    train_tokens = tokens[: split_point + seq_len + 1]
    val_tokens = tokens[split_point:]

    train_dataset = TextLMDataset(train_tokens, seq_len, num_samples, seed=seed)
    val_dataset = TextLMDataset(val_tokens, seq_len, val_samples, seed=seed + 1)
    return train_dataset, val_dataset


def get_dataloaders(num_samples: int, val_samples: int, seq_len: int, vocab_size: int, batch_size: int, seed: int = 42):
    """
    Returns train and validation dataloaders using Tiny Shakespeare.
    pin_memory=True pre-pins CPU tensors for faster host→GPU transfers.
    num_workers loads batches in background processes so the GPU never stalls waiting for data.
    """
    use_cuda = torch.cuda.is_available()

    cache_dir = os.path.join(os.path.dirname(__file__), "dataset_cache")
    train_dataset, val_dataset = load_tiny_shakespeare(cache_dir, seq_len, num_samples, val_samples, seed=seed)

    num_workers = min(4, (os.cpu_count() or 1) // 2) if use_cuda else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=use_cuda and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=num_workers,
        persistent_workers=use_cuda and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2,
    )

    return train_loader, val_loader


def decode_tokens(tokens: list) -> str:
    """
    Decodes a list of token IDs back into characters.
    """
    return "".join([chr(t) if 0 <= t < 128 else "?" for t in tokens])
