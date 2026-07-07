import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
import logging

# Silence standard Qiskit ML gradient warnings to keep training logs clean
logging.getLogger("qiskit_machine_learning").setLevel(logging.ERROR)

def build_quantum_circuit(num_qubits: int, q_depth: int, entangler: str = "ring"):
    """
    Constructs a Parameterized Quantum Circuit (PQC).
    - Encoding: Angle embedding via Ry gates.
    - Ansatz: Parameterized Rx and Ry gates followed by CX entangling gates in a 1D chain.
    """
    inputs = ParameterVector("x", num_qubits)
    # Number of variational parameters per layer = 2 * num_qubits (for Rx and Ry)
    # Total parameters = 2 * num_qubits * q_depth
    weights = ParameterVector("w", 2 * num_qubits * q_depth)
    
    qc = QuantumCircuit(num_qubits)
    
    # 1. Feature Map (Angle Encoding)
    # Encodes classical inputs (bounded or scaled features) as qubit rotation angles
    for i in range(num_qubits):
        qc.ry(inputs[i], i)
        
    # 2. Variational Ansatz Layers
    param_idx = 0
    for d in range(q_depth):
        # Single-qubit rotations
        for i in range(num_qubits):
            qc.rx(weights[param_idx], i)
            qc.ry(weights[param_idx + 1], i)
            param_idx += 2
        
        # Entanglement layer
        # - ring: nearest-neighbor chain with closing link (original behavior)
        # - full: entangle every pair (denser, stronger entanglement)
        # - linear: simple chain without closing link
        if entangler == "ring":
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            if num_qubits > 2:
                qc.cx(num_qubits - 1, 0)
        elif entangler == "linear":
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        elif entangler == "full":
            # full entanglement: CX between all pairs (i < j)
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qc.cx(i, j)
        else:
            # fallback to ring
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            if num_qubits > 2:
                qc.cx(num_qubits - 1, 0)
            
    return qc, inputs, weights

def get_estimator(use_noisy_simulator: bool, shots: int = 1024, depol_error: float = 0.01):
    """
    Instantiates a Qiskit V2 primitive Estimator.
    - If use_noisy_simulator is False: Returns exact StatevectorEstimator (noise-free).
      Uses max_workers equal to CPU count so all circuits in a batch run in parallel
      threads, giving the best throughput on a multi-core machine.
    - If use_noisy_simulator is True: Returns Aer EstimatorV2 with shots and depolarizing noise.
    """
    if not use_noisy_simulator:
        return StatevectorEstimator()
    else:
        # Build depolarizing noise model
        noise_model = NoiseModel()
        error_1 = depolarizing_error(depol_error, 1)
        noise_model.add_all_qubit_quantum_error(error_1, ["rx", "ry"])
        
        # 2-qubit gate error (cx) is double the 1-qubit gate error
        error_2 = depolarizing_error(depol_error * 2, 2)
        noise_model.add_all_qubit_quantum_error(error_2, ["cx"])
        
        options = {
            "run_options": {"shots": shots},
            "backend_options": {"noise_model": noise_model}
        }
        return EstimatorV2(options=options)

class QuantumFeatureExtractor(nn.Module):
    """
    A PyTorch wrapper around Qiskit's EstimatorQNN/TorchConnector or a GPU-native
    approximation for quantum feature extraction.
    Takes 3D sequence tensors of shape [batch_size, seq_len, num_qubits],
    flattens them to 2D for the quantum connector or GPU module, and restores the 3D shape.
    """
    def __init__(self, num_qubits: int, q_depth: int, use_noisy_simulator: bool = False,
                 shots: int = 1024, depol_error: float = 0.01, use_gpu_simulator: bool = False,
                 entangler: str = "ring"):
        super().__init__()
        self.num_qubits = num_qubits
        self.q_depth = q_depth
        self.use_gpu_simulator = use_gpu_simulator and torch.cuda.is_available()
        self.quantum_scale = nn.Parameter(torch.tensor(0.1))

        if self.use_gpu_simulator:
            hidden_dim = max(num_qubits * 16, 64)
            self.gpu_qnn = nn.Sequential(
                nn.Linear(num_qubits, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_qubits),
                nn.Tanh()
            )
            self.qnn_layer = None
            self.qnn = None
            self.estimator = None
        else:
            # Build circuit with requested entangler
            self.qc, self.inputs, self.weights = build_quantum_circuit(num_qubits, q_depth, entangler=entangler)
            
            # Build observables: measure Z expectation value on each qubit
            self.observables = []
            for q in range(num_qubits):
                pauli_list = ["I"] * num_qubits
                pauli_list[q] = "Z"
                # Reverse because Qiskit labels qubits from right to left
                pauli_str = "".join(reversed(pauli_list))
                self.observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
                
            # Get estimator primitive
            self.estimator = get_estimator(use_noisy_simulator, shots, depol_error)
            
            # Create Qiskit QNN
            self.qnn = EstimatorQNN(
                circuit=self.qc,
                input_params=self.inputs,
                weight_params=self.weights,
                observables=self.observables,
                estimator=self.estimator,
                input_gradients=True
            )
            
            # Connect to PyTorch
            # Initial weights can be random or small uniform values
            initial_weights = torch.randn(self.qnn.num_weights) * 0.1
            self.qnn_layer = TorchConnector(self.qnn, initial_weights=initial_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, num_qubits]
        batch_size, seq_len, features = x.shape
        assert features == self.num_qubits, f"Expected {self.num_qubits} features, got {features}"

        if self.use_gpu_simulator:
            x_flat = x.contiguous().view(-1, self.num_qubits)
            out_flat = self.gpu_qnn(x_flat)
            out_flat = x_flat + self.quantum_scale * out_flat
            out = out_flat.view(batch_size, seq_len, self.num_qubits)
            return out

        # Remember the original device (may be CUDA) so we can move results back.
        # Qiskit's TorchConnector/EstimatorQNN only supports CPU execution.
        original_device = x.device

        # Flatten sequence dimension to process each token vector independently
        # shape: [batch_size * seq_len, num_qubits]
        x_flat = x.contiguous().view(-1, self.num_qubits)
        
        # Scale inputs to [-pi, pi] to match rotation angle limits
        # Using tanh projection to bound the raw classical values gracefully
        x_scaled = torch.tanh(x_flat) * torch.pi

        # Move to CPU for Qiskit (no-op if already on CPU)
        x_cpu = x_scaled.cpu()

        # Run QNN forward pass (always on CPU)
        out_cpu = self.qnn_layer(x_cpu)

        # Apply residual correction relative to the input encoding
        out_flat = x_cpu + self.quantum_scale * out_cpu

        # Move result back to the original device (GPU if CUDA is being used)
        out_flat = out_flat.to(original_device)

        # Restore shape to [batch_size, seq_len, num_qubits]
        out = out_flat.view(batch_size, seq_len, self.num_qubits)
        return out
