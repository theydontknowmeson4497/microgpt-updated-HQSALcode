import os

class Hyperparameters:
    # Reproducibility
    seed = 42

    # Data Settings
    vocab_size = 12       # Small synthetic vocab (e.g., characters or small words)
    seq_len = 6           # REDUCED: Shorter sequences = faster attention computation
    num_samples = 16      # Number of training samples (reduced for fast local runs)
    val_samples = 4       # Number of validation samples (reduced for fast local runs)

    # Model Settings
    embed_dim = 12        # REDUCED: Smaller embeddings = fewer computations
    num_heads = 4         # Total attention heads
    # Number of quantum heads (must be <= num_heads). 
    # Hybrid setup: Head 0 = classical, Heads 1,2,3 = quantum (total 3 quantum heads)
    num_quantum_heads = 3 
    
    # Qubit count per quantum head. Maps query/key projection size to qubits.
    num_qubits = 3        # REDUCED: Fewer qubits = exponentially faster quantum simulation
    q_depth = 1           # Depth of variational ansatz circuit (reduced for fast gradients)

    ffn_dim = 24          # REDUCED: Smaller FFN = fewer parameters to update
    num_layers = 1        # Number of transformer decoder layers
    dropout = 0.0         # REDUCED: No dropout speeds up forward passes slightly

    # Training Settings
    batch_size = 8        # INCREASED: Larger batch (if memory allows) uses more vectorization
    learning_rate = 0.02  # INCREASED: Higher LR for faster convergence
    epochs = 3
    device = "cpu"        # "cpu" is preferred for PyTorch + Qiskit interop locally

    # Quantum Simulator Settings
    use_noisy_simulator = False # Keep False for maximum speed
    shots = 256           # REDUCED: Fewer shots = faster noisy simulation (if enabled)
    depolarizing_error = 0.01
