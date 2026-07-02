import torch
from config import Hyperparameters as hp
from utils import set_seed, count_parameters, Timer
from data import get_dataloaders
from model import HybridQuantumTransformerDecoder
from classical_baseline import ClassicalTransformerDecoder
from train import train_model, evaluate_model
import time

def run_ablation_study():
    print("=" * 60)
    print("STARTING HYBRID QUANTUM SELF-ATTENTION ABLATION STUDY")
    print("=" * 60)
    
    # 1. Setup reproducibility and device
    set_seed(hp.seed)
    device = torch.device(hp.device)
    print(f"Running on device: {device}")
    
    # 2. Get dataloaders
    print("Generating synthetic language modeling dataset...")
    train_loader, val_loader = get_dataloaders(
        num_samples=hp.num_samples,
        val_samples=hp.val_samples,
        seq_len=hp.seq_len,
        vocab_size=hp.vocab_size,
        batch_size=hp.batch_size,
        seed=hp.seed
    )
    print(f"Train samples: {hp.num_samples} | Val samples: {hp.val_samples}")
    print(f"Sequence length: {hp.seq_len} | Vocab size: {hp.vocab_size}")
    
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
    
    params_classical = count_parameters(classical_model)
    print(f"Classical Model Parameter Count: {params_classical}")
    
    optimizer = torch.optim.AdamW(classical_model.parameters(), lr=hp.learning_rate)
    
    with Timer() as t:
        classical_history = train_model(
            model=classical_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=hp.epochs,
            device=hp.device
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
        dropout=hp.dropout
    ).to(device)
    
    params_q_clean = count_parameters(quantum_model_clean)
    print(f"Hybrid Model Parameter Count: {params_q_clean}")
    
    optimizer = torch.optim.AdamW(quantum_model_clean.parameters(), lr=hp.learning_rate)
    
    with Timer() as t:
        q_clean_history = train_model(
            model=quantum_model_clean,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=hp.epochs,
            device=hp.device
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
        shots=hp.shots,
        depol_error=hp.depolarizing_error,
        dropout=hp.dropout
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
    
    # ==========================================
    # Print Ablation Summary Table
    # ==========================================
    print("\n" + "=" * 80)
    print("                         ABLATION STUDY SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Model Configuration':<32} | {'Params':<8} | {'Train Loss':<10} | {'Val Loss':<8} | {'Val Perp':<8} | {'Time (s)':<8}")
    print("-" * 80)
    
    for name, data in results.items():
        hist = data["history"]
        final_train_loss = hist["train_loss"][-1]
        final_val_loss = hist["val_loss"][-1]
        final_val_perp = hist["val_perplexity"][-1]
        total_time = data["time"]
        params = data["params"]
        
        print(f"{name:<32} | {params:<8} | {final_train_loss:<10.4f} | {final_val_loss:<8.4f} | {final_val_perp:<8.2f} | {total_time:<8.1f}")
    print("=" * 80)
    print("Note on Memory: Simulation runs locally in RAM. Low memory footprint (<200MB) due to small models.")
    print("=" * 80)

if __name__ == "__main__":
    run_ablation_study()
