import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Define global vocabulary
# Vocab size = 12
# 0: <pad>, 1: <sos>, 2: <eos>, 3-11: letters A-I
VOCAB = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "A": 3,
    "B": 4,
    "C": 5,
    "D": 6,
    "E": 7,
    "F": 8,
    "G": 9,
    "H": 10,
    "I": 11
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

def generate_sequence(seq_len: int, vocab_size: int) -> list:
    """
    Generates a single synthetic sequence of a repeating pattern.
    Example: [sos, A, B, C, A, B, C, ..., eos]
    This forces the self-attention layer to learn to look back to previous symbols.
    """
    seq = [VOCAB["<sos>"]]
    
    # Choose a pattern length between 2 and 4
    pattern_len = random.randint(2, 4)
    # Sample random letters from A to I (IDs 3 to vocab_size-1)
    max_vocab_idx = min(11, vocab_size - 1)
    pattern = [random.randint(3, max_vocab_idx) for _ in range(pattern_len)]
    
    # Fill the sequence with the repeating pattern
    while len(seq) < seq_len - 1:
        seq.append(pattern[(len(seq) - 1) % pattern_len])
        
    seq.append(VOCAB["<eos>"])
    return seq[:seq_len]

class SyntheticLMDataset(Dataset):
    """
    A PyTorch dataset for synthetic autoregressive language modeling.
    For each sequence x, the target y is x shifted by 1.
    """
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.data = []
        for _ in range(num_samples):
            self.data.append(generate_sequence(seq_len, vocab_size))
            
        self.data = torch.tensor(self.data, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Autoregressive setup:
        # Input: tokens 0 to seq_len-2
        # Target: tokens 1 to seq_len-1
        seq = self.data[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y

def get_dataloaders(num_samples: int, val_samples: int, seq_len: int, vocab_size: int, batch_size: int, seed: int = 42):
    """
    Returns train and validation dataloaders.
    """
    train_dataset = SyntheticLMDataset(num_samples, seq_len, vocab_size, seed=seed)
    val_dataset = SyntheticLMDataset(val_samples, seq_len, vocab_size, seed=seed + 1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def decode_tokens(tokens: list) -> str:
    """
    Decodes a list of token IDs back into characters.
    """
    return " ".join([INV_VOCAB.get(t, "?") for t in tokens])
