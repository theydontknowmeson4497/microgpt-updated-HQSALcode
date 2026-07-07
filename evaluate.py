import torch
from config import Hyperparameters as hp
from utils import set_seed, count_parameters, Timer, get_device
from data import get_dataloaders
from model import HybridQuantumTransformerDecoder
from classical_baseline import ClassicalTransformerDecoder
from train import train_model, evaluate_model
import time


def build_optimizer_and_scheduler(model: torch.nn.Module):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
    )
    # Use cosine annealing for smoother long-run convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hp.epochs,
        eta_min=hp.min_lr
    )
    return optimizer, scheduler


def print_summary(results: dict):
    print("\n" + "=" * 80)
    print("                         ABLATION STUDY SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Model Configuration':<32} | {'Params':<10} | {'Train Loss':<10} | {'Val Loss':<8} | {'Val Perp':<8} | {'Time (s)':<8}")
    print("-" * 80)

    best_model = None
    best_val_loss = float('inf')

    for name, data in results.items():
        hist = data["history"]
        final_train_loss = hist["train_loss"][-1] if hist["train_loss"] else float('nan')
        final_val_loss = hist["val_loss"][-1] if hist["val_loss"] else float('nan')
        final_val_perp = hist["val_perplexity"][-1] if hist["val_perplexity"] else float('nan')
        total_time = data["time"]
        params = data["params"]

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_model = name

        print(f"{name:<32} | {params:<10} | {final_train_loss:<10.4f} | {final_val_loss:<8.4f} | {final_val_perp:<8.2f} | {total_time:<8.1f}")

    print("=" * 80)
    if best_model is not None:
        print(f"Best validation result: {best_model} with val loss {best_val_loss:.4f}")
    print("Note: Noisy hybrid evaluation is performed only in inference mode using the trained noise-free weights.")
    print("=" * 80)


def run_ablation_study():
    print("=" * 60)
    print("STARTING HYBRID QUANTUM SELF-ATTENTION ABLATION STUDY")
    print("=" * 60)
    
    # 1. Setup reproducibility and device
    set_seed(hp.seed)
    device = get_device()          # prints GPU name or CPU fallback message
    print(f"Running on device: {device}")
    
    # 2. Get dataloaders
    print("Loading Tiny Shakespeare language modeling dataset...")
    train_loader, val_loader = get_dataloaders(
        num_samples=hp.num_samples,
        val_samples=hp.val_samples,
        seq_len=hp.seq_len,
        vocab_size=hp.vocab_size,
        batch_size=hp.batch_size,
        seed=hp.seed
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")
    print(f"Sequence length: {hp.seq_len} | Vocabulary size: {hp.vocab_size}")
    
    results = {}
    
    # ==========================================
    # Run 1: Classical Baseline
    # ==========================================
    print("\n" + "-" * 50)
    print("1. TRAINING CLASSICAL-ONLY BASELINE MODEL")
    print("-" * 50)
    
    set_seed(hp.seed) # Reset seed for fair starting weight initialization
    classical_model = ClassicalTransformerDecoder(
        vocab_size=hp.vocab_size,
        embed_dim=hp.embed_dim,
        seq_len=hp.seq_len,
        num_heads=hp.num_heads,
        ffn_dim=hp.ffn_dim,
        num_layers=hp.num_layers,
        dropout=hp.dropout
    ).to(device)

    # Do not use torch.compile in this environment because it may require
    # Triton backend modules that are not available or fully functional.
    # The model still uses AMP and GPU-optimized cuDNN settings for speed.
    params_classical = count_parameters(classical_model)
    print(f"Classical Model Parameter Count: {params_classical}")
    optimizer, scheduler = build_optimizer_and_scheduler(classical_model)

    with Timer() as t:
        classical_history = train_model(
            model=classical_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=hp.epochs,
            device=device,
            patience=hp.patience,
            min_lr=hp.min_lr
        )
    classical_time = t.elapsed

    results["Classical Baseline"] = {
        "params": params_classical,
        "history": classical_history,
        "time": classical_time
    }
    
    # ==========================================
    # Run 2: Hybrid Quantum Attention (Noise-Free)
    # ==========================================
    print("\n" + "-" * 50)
    print("2. TRAINING HYBRID QUANTUM ATTENTION (NOISE-FREE SIMULATOR)")
    print("-" * 50)
    
    set_seed(hp.seed) # Reset seed
    quantum_model_clean = HybridQuantumTransformerDecoder(
        vocab_size=hp.vocab_size,
        embed_dim=hp.embed_dim,
        seq_len=hp.seq_len,
        num_heads=hp.num_heads,
        num_quantum_heads=hp.num_quantum_heads,
        num_qubits=hp.num_qubits,
        q_depth=hp.q_depth,
        ffn_dim=hp.ffn_dim,
        num_layers=hp.num_layers,
        use_noisy_simulator=False,
        use_gpu_simulator=(device.type == "cuda"),
        dropout=hp.dropout,
        entangler=hp.entangler
    ).to(device)
    
    params_q_clean = count_parameters(quantum_model_clean)
    print(f"Hybrid Model Parameter Count: {params_q_clean}")
    
    optimizer, scheduler = build_optimizer_and_scheduler(quantum_model_clean)
    
    with Timer() as t:
        q_clean_history = train_model(
            model=quantum_model_clean,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=hp.epochs,
            device=device,
            patience=hp.patience,
            min_lr=hp.min_lr
        )
    q_clean_time = t.elapsed
    
    results["Hybrid Quantum (Noise-Free)"] = {
        "params": params_q_clean,
        "history": q_clean_history,
        "time": q_clean_time
    }
    
    # ==========================================
    # Run 3: Hybrid Quantum Attention (Noisy Simulator Evaluation)
    # ==========================================
    print("\n" + "-" * 50)
    print("3. EVALUATING HYBRID QUANTUM ATTENTION (NOISY / SHOT-BASED SIMULATOR)")
    print("-" * 50)
    
    set_seed(hp.seed) # Reset seed
    quantum_model_noisy = HybridQuantumTransformerDecoder(
        vocab_size=hp.vocab_size,
        embed_dim=hp.embed_dim,
        seq_len=hp.seq_len,
        num_heads=hp.num_heads,
        num_quantum_heads=hp.num_quantum_heads,
        num_qubits=hp.num_qubits,
        q_depth=hp.q_depth,
        ffn_dim=hp.ffn_dim,
        num_layers=hp.num_layers,
        use_noisy_simulator=True,
        use_gpu_simulator=(device.type == "cuda"),
        shots=hp.shots,
        depol_error=hp.depolarizing_error,
        dropout=hp.dropout,
        entangler=hp.entangler
    ).to(device)
    
    # Load the weights from the trained noise-free model
    quantum_model_noisy.load_state_dict(quantum_model_clean.state_dict())
    
    params_q_noisy = count_parameters(quantum_model_noisy)
    print(f"Noisy Hybrid Model Parameter Count: {params_q_noisy}")
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    with Timer() as t:
        val_loss, val_perp = evaluate_model(quantum_model_noisy, val_loader, criterion, device)
    q_noisy_time = t.elapsed
    
    # Mock history for output reporting
    q_noisy_history = {
        "train_loss": [q_clean_history["train_loss"][-1]], # Use clean train loss as approximation
        "val_loss": [val_loss],
        "val_perplexity": [val_perp]
    }
    
    results["Hybrid Quantum (Noisy Inference)"] = {
        "params": params_q_noisy,
        "history": q_noisy_history,
        "time": q_noisy_time
    }
    
    print_summary(results)

if __name__ == "__main__":
    run_ablation_study()
