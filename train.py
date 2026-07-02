import torch
import torch.nn as nn
from utils import Timer
import math

def train_one_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                    device: str) -> float:
    """
    Trains the model for one epoch.
    Returns average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        # logits shape: [batch_size, seq_len, vocab_size]
        logits, _ = model(x)
        
        # Flatten logits and targets for CrossEntropyLoss
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (critical for quantum layers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step optimizer
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
        # Visual batch feedback for quantum simulation runs
        if "Hybrid" in model.__class__.__name__:
            print(f"    -> Batch {batch_idx + 1}/{num_batches} complete (Loss: {loss.item():.4f})", flush=True)
            
    return total_loss / len(dataloader.dataset)

def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                   criterion: nn.Module, device: str) -> tuple:
    """
    Evaluates the model on validation data.
    Returns average loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.size(0)
            
    avg_loss = total_loss / len(dataloader.dataset)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')
        
    return avg_loss, perplexity

def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                epochs: int, device: str) -> dict:
    """
    Runs the full multi-epoch training and validation loop.
    Returns training and validation loss history.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <pad> token
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "epoch_times": []
    }
    
    for epoch in range(epochs):
        with Timer() as t:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_perp = evaluate_model(model, val_loader, criterion, device)
            
        epoch_time = t.elapsed
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_perp)
        history["epoch_times"].append(epoch_time)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {val_perp:.2f} | "
              f"Time: {epoch_time:.2f}s")
              
    return history
