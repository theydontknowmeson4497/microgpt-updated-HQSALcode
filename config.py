import os

class Hyperparameters:
    # Reproducibility
    seed = 42

    # Data Settings
    vocab_size = 12       # Small synthetic vocab (e.g., characters or small words)
    seq_len = 8           # Mini sequence length for fast local laptop simulation (supports up to 256)
    num_samples = 16      # Number of training samples (reduced for fast local runs)
    val_samples = 4       # Number of validation samples (reduced for fast local runs)

    # Model Settings
    embed_dim = 16        # Embedding dimension
    num_heads = 2         # Total attention heads
    # Number of quantum heads (must be <= num_heads). 
    # For a hybrid block, we can have 1 quantum head and 1 classical head.
    num_quantum_heads = 1 
    
    # Qubit count per quantum head. Maps query/key projection size to qubits.
    num_qubits = 4        
    q_depth = 1           # Depth of variational ansatz circuit (reduced for fast gradients)

    ffn_dim = 32          # FFN hidden layer dimension
    num_layers = 1        # Number of transformer decoder layers
    dropout = 0.1         # Dropout rate

    # Training Settings
    batch_size = 4
    learning_rate = 0.01
    epochs = 3
    device = "cpu"        # "cpu" is preferred for PyTorch + Qiskit interop locally

    # Quantum Simulator Settings
    use_noisy_simulator = False # Toggle noisy simulator in evaluation
    shots = 1024                # Number of shots for noisy measurement
    depolarizing_error = 0.01   # Depolarizing noise probability for shot-based simulation
