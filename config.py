import os
import torch

class Hyperparameters:
    # Reproducibility
    seed = 42

    # Data Settings
    vocab_size = 32033    # ~32k unique words from input.txt + 1 BOS token (set at runtime)
    seq_len = 3           # BOS + word + BOS per training example
    num_samples = 16
    val_samples = 4
    # Max training steps — all 32k words are in vocab but we cap steps so the
    # classical scalar-autograd engine does not OOM on 16 GB RAM.
    # Each step builds a Python-object computation graph; 2000 steps is safe.
    max_train_steps = 2000

    # Model Settings
    # embed_dim drives memory in the classical model quadratically (embed_dim^2 per
    # attention weight matrix, all stored as individual Python Value objects).
    # 64 keeps the graph manageable on 16 GB while still giving a meaningful model.
    embed_dim = 64
    num_heads = 4
    # Original: 3 quantum heads, 1 classical head
    num_quantum_heads = 3

    # Qubit count per quantum head — fixed small number regardless of head_dim
    num_qubits = 4
    q_depth = 1

    ffn_dim = 128
    num_layers = 1
    dropout = 0.0

    # Training Settings
    batch_size = 1
    learning_rate = 0.01
    epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quantum Simulator Settings
    use_noisy_simulator = False # Keep False for maximum speed
    shots = 256           # Fewer shots for faster noisy simulation (if enabled)
    depolarizing_error = 0.01
