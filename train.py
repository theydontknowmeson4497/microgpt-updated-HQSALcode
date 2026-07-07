import torch
import torch.nn as nn
from utils import Timer
import math

def train_one_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                    device: torch.device) -> float:
    """
    Trains the model for one epoch.
    Returns average training loss.
    Uses automatic mixed precision on CUDA for faster GPU throughput.
    """
    model.train()
    total_loss = 0.0

    use_amp = device.type == "cuda" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)

def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                   criterion: nn.Module, device: str) -> tuple:
    """
    Evaluates the model on validation data.
    Returns average loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    use_amp = str(device) != "cpu" and torch.cuda.is_available()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
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
                scheduler: torch.optim.lr_scheduler._LRScheduler | None, 
                epochs: int, device: str, patience: int = 5, min_lr: float = 1e-6) -> dict:
    """
    Runs the full multi-epoch training and validation loop.
    Returns training and validation loss history.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "epoch_times": []
    }

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        with Timer() as t:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_perp = evaluate_model(model, val_loader, criterion, device)

        epoch_time = t.elapsed

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_perp)
        history["epoch_times"].append(epoch_time)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Perplexity: {val_perp:.2f} | "
              f"LR: {current_lr:.6g} | "
              f"Time: {epoch_time:.2f}s")

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs (no improvement for {patience} epochs).")
            break

        if optimizer.param_groups[0]["lr"] <= min_lr:
            print(f"Learning rate has reached the minimum threshold ({min_lr}). Stopping training.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
