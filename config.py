import os
import torch

class Hyperparameters:
    # Reproducibility
    seed = 42
    # Multiple seeds for robust experiments
    seeds = [42, 123, 256, 512, 1024]

    # Data Settings
    vocab_size = 128      # Byte-level vocabulary for Tiny Shakespeare text
    seq_len = 64          # Longer sequences for real language modeling
    num_samples = 10000   # Larger training sample count for hybrid learning
    val_samples = 2000    # Validation sample count for meaningful evaluation

    # Model Settings
    embed_dim = 256       # Larger model capacity for stronger hybrid comparison
    num_heads = 4         # Total attention heads
    num_quantum_heads = 1 # One quantum head improves stability

    # Stronger hybrid approximation on GPU: larger qubit embedding and deeper processing
    num_qubits = 6
    q_depth = 3

    ffn_dim = 512         # Large FFN = heavy GPU work during classical phase
    num_layers = 2
    dropout = 0.0

    # Training Settings — smaller learning rate, longer schedule, weight decay
    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 0.01
    epochs = 50
    patience = 5
    min_lr = 1e-6

    # Device: auto-detect CUDA (RTX GPU), fall back to CPU.
    # Classical PyTorch layers run on GPU. Qiskit quantum circuits always run on CPU
    # internally — the device bridge in quantum_layer.py handles the transfer.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quantum Simulator Settings
    use_noisy_simulator = False  # Keep False for maximum speed
    shots = 256
    depolarizing_error = 0.01
    # Quantum circuit entangler type: 'ring', 'linear', or 'full'
    entangler = 'full'
