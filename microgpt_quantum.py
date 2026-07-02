"""
microgpt_quantum.py

A PyTorch port of Andrej Karpathy's microGPT where exactly ONE attention head (Head 0)
is replaced by a Parameterized Quantum Circuit (VQC) via Qiskit Machine Learning.

========================================================================================
QUANTUM HEAD AND GRADIENT FLOW DESCRIPTION:
- The Quantum Head is implemented within the `QuantumGPT` class inside the `forward` pass.
- In `forward`, Query (Q) and Key (K) slices of Head 0 (size 4) are classically projected 
  using Linear(4 -> 4) layers, scaled to [-pi, pi] using `tanh(x) * pi` to fit within the 
  qubit rotation bounds, and passed through the VQC wrapper `vqc` (Qiskit TorchConnector).
- The VQC uses a 4-qubit Angle Embedding (RY rotations) as the feature map, followed by 
  a parameterized ansatz (RX + RY rotations, CNOT ring) and Pauli-Z expectation measurements.
- For backpropagation, we set `input_gradients=True` in EstimatorQNN. This triggers the 
  analytical Parameter-Shift Rule w.r.t both the circuit weights AND the input features:
      ∂f/∂x = [f(x + π/2) - f(x - π/2)] / 2
- By computing the analytical input gradients w.r.t the feature map inputs, the VQC acts 
  as a fully differentiable node in PyTorch's computation graph. The gradients flow back 
  through the input projection layers, and ultimately update the preceding token and 
  positional embedding layers, allowing end-to-end training of the hybrid system.
========================================================================================
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator

class RMSNorm(nn.Module):
    """Parameter-free RMSNorm exactly matching microGPT's definition."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)

class QuantumGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=16, block_size=16, n_head=4, n_layer=1, use_noisy=False):
        super().__init__()
        torch.manual_seed(42)
        random.seed(42)
        
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_layer = n_layer
        
        # Embedding Layers
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        
        # Attention Projection Weights
        self.attn_wq = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wk = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wv = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wo = nn.Linear(n_embd, n_embd, bias=False)
        
        # MLP Layers
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        
        # LM Head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Classical Projection Adapters for Quantum Head (Head 0)
        self.q_in_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.q_out_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_in_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_out_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Build 4-Qubit Parameterized Quantum Circuit (PQC)
        self.inputs = ParameterVector("x", self.head_dim)
        self.weights = ParameterVector("w", 2 * self.head_dim) # 8 trainable weights
        
        self.qc = QuantumCircuit(self.head_dim)
        # Angle Embedding Feature Map
        for i in range(self.head_dim):
            self.qc.ry(self.inputs[i], i)
        # Parameterized Ansatz Rotations
        for i in range(self.head_dim):
            self.qc.rx(self.weights[2*i], i)
            self.qc.ry(self.weights[2*i+1], i)
        # Entangling Ring of CNOTs
        for i in range(self.head_dim - 1):
            self.qc.cx(i, i+1)
        self.qc.cx(self.head_dim - 1, 0)
        
        # Observables (Z expectation value on all qubits)
        self.observables = []
        for q in range(self.head_dim):
            pauli_list = ["I"] * self.head_dim
            pauli_list[q] = "Z"
            pauli_str = "".join(reversed(pauli_list))
            self.observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
            
        # Estimator Setup
        self.estimator = self._get_estimator(use_noisy)
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=self.inputs,
            weight_params=self.weights,
            observables=self.observables,
            estimator=self.estimator,
            input_gradients=True # Essential for backprop to embedding layers
        )
        self.vqc = TorchConnector(self.qnn)
        
        # RMSNorm blocks
        self.rmsnorm1 = RMSNorm(n_embd)
        self.rmsnorm2 = RMSNorm(n_embd)
        self.rmsnorm3 = RMSNorm(n_embd)
        
        # Gaussian Initialization (std=0.08) matching microGPT
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.08)

    def _get_estimator(self, use_noisy, shots=1024, depol_error=0.01):
        if not use_noisy:
            return StatevectorEstimator()
        else:
            from qiskit_aer.primitives import EstimatorV2
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            noise_model = NoiseModel()
            error_1 = depolarizing_error(depol_error, 1)
            noise_model.add_all_qubit_quantum_error(error_1, ["rx", "ry"])
            error_2 = depolarizing_error(depol_error * 2, 2)
            noise_model.add_all_qubit_quantum_error(error_2, ["cx"])
            options = {
                "run_options": {"shots": shots},
                "backend_options": {"noise_model": noise_model}
            }
            return EstimatorV2(options=options)

    def set_noisy_simulator(self, use_noisy, shots=1024, depol_error=0.01):
        """Swaps the Qiskit Estimator to a noisy AerSimulator configuration."""
        old_weights = self.vqc.weight.data.clone()
        self.estimator = self._get_estimator(use_noisy, shots, depol_error)
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=self.inputs,
            weight_params=self.weights,
            observables=self.observables,
            estimator=self.estimator,
            input_gradients=True
        )
        self.vqc = TorchConnector(self.qnn, initial_weights=old_weights)

    def forward(self, token_id, pos_id, keys_cache, values_cache, q_keys_cache):
        # Embeddings
        tok_emb = self.wte(token_id).squeeze(0)
        pos_emb = self.wpe(torch.tensor([pos_id], device=token_id.device)).squeeze(0)
        x = tok_emb + pos_emb
        x = self.rmsnorm1(x)
        
        # Layer 0 Attention block
        x_residual = x
        x = self.rmsnorm2(x)
        
        q = self.attn_wq(x)
        k = self.attn_wk(x)
        v = self.attn_wv(x)
        
        # Update KV cache (requires dynamic concatenation for backpropagation)
        keys_cache[0] = torch.cat([keys_cache[0], k.unsqueeze(0)], dim=0)
        values_cache[0] = torch.cat([values_cache[0], v.unsqueeze(0)], dim=0)
        
        head_outputs = []
        
        for h in range(self.n_head):
            hs = h * self.head_dim
            q_h = q[hs : hs+self.head_dim]
            k_h = keys_cache[0][pos_id, hs : hs+self.head_dim]
            v_h = values_cache[0][:, hs : hs+self.head_dim]
            
            if h == 0:
                # --- Quantum Head ---
                # 1. Project Query and scale
                q_h_proj = self.q_in_proj(q_h.unsqueeze(0))
                q_h_scaled = torch.tanh(q_h_proj) * torch.pi
                q_h_vqc = self.vqc(q_h_scaled)
                q_h_prime = self.q_out_proj(q_h_vqc).squeeze(0)
                
                # 2. Project Key and scale
                k_h_proj = self.k_in_proj(k_h.unsqueeze(0))
                k_h_scaled = torch.tanh(k_h_proj) * torch.pi
                k_h_vqc = self.vqc(k_h_scaled)
                k_h_prime = self.k_out_proj(k_h_vqc)
                
                # 3. Cache Quantum Key
                q_keys_cache[0] = torch.cat([q_keys_cache[0], k_h_prime], dim=0)
                
                # 4. Dot-product attention scores
                attn_logits = torch.matmul(q_h_prime.unsqueeze(0), q_keys_cache[0].transpose(0, 1)).squeeze(0) / math.sqrt(self.head_dim)
                attn_weights = torch.softmax(attn_logits, dim=-1)
                head_out = torch.matmul(attn_weights.unsqueeze(0), v_h).squeeze(0)
            else:
                # --- Classical Head ---
                k_h_cache = keys_cache[0][:, hs : hs+self.head_dim]
                attn_logits = torch.matmul(q_h.unsqueeze(0), k_h_cache.transpose(0, 1)).squeeze(0) / math.sqrt(self.head_dim)
                attn_weights = torch.softmax(attn_logits, dim=-1)
                head_out = torch.matmul(attn_weights.unsqueeze(0), v_h).squeeze(0)
                
            head_outputs.append(head_out)
            
        x_attn = torch.cat(head_outputs, dim=-1)
        x = self.attn_wo(x_attn)
        x = x + x_residual
        
        # MLP block
        x_residual = x
        x = self.rmsnorm3(x)
        x = self.mlp_fc1(x)
        x = torch.relu(x)
        x = self.mlp_fc2(x)
        x = x + x_residual
        
        logits = self.lm_head(x)
        return logits

def train_and_evaluate(use_noisy_eval=False, num_steps=1000):
    print("=" * 60)
    print("STARTING HYBRID QUANTUM MICROGPT TRAINING")
    print("=" * 60)
    
    torch.manual_seed(42)
    random.seed(42)
    
    # 1. Dataset loading
    if not os.path.exists('input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, 'input.txt')
    docs = [line.strip() for line in open('input.txt') if line.strip()]
    random.shuffle(docs)

    # 2. Tokenizer setup
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1

    # 3. Model instantiation
    model = QuantumGPT(vocab_size=vocab_size, use_noisy=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_params = model.vqc.weight.numel()
    print(f"Total Params: {total_params} | Quantum Params: {quantum_params}")

    # 4. Optimizer and training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    # num_steps parameter passed to function
    all_step_losses = []
    
    start_time = time.perf_counter()
    
    # Run training (always noise-free for local efficiency)
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(model.block_size, len(tokens) - 1)

        # Initialize KV Cache and Quantum Key cache
        keys_cache = [torch.zeros(0, model.n_embd)]
        values_cache = [torch.zeros(0, model.n_embd)]
        q_keys_cache = [torch.zeros(0, model.head_dim)]
        
        optimizer.zero_grad()
        
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            token_tensor = torch.tensor([token_id], dtype=torch.long)
            target_tensor = torch.tensor([target_id], dtype=torch.long)
            
            logits = model(token_tensor, pos_id, keys_cache, values_cache, q_keys_cache)
            loss_t = criterion(logits.unsqueeze(0), target_tensor)
            losses.append(loss_t)
            
        loss = sum(losses) / n
        all_step_losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        
        # Adam optimizer update with linear LR decay
        lr_t = 0.01 * (1 - step / num_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_t
            
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            print(f"Step {step+1:4d}/{num_steps:4d} | Loss: {loss.item():.4f}")

    training_time = time.perf_counter() - start_time
    print(f"Finished training in {training_time:.2f} seconds.")

    # 5. Optional Noisy Evaluation
    noisy_val_loss = "N/A"
    if use_noisy_eval:
        print("\nSwitching to noisy AerSimulator (1% depolarizing noise, 1024 shots) for evaluation...")
        model.set_noisy_simulator(use_noisy=True, shots=1024, depol_error=0.01)
        model.eval()
        
        # Evaluate on next 50 validation documents to report average noisy loss
        val_losses = []
        with torch.no_grad():
            for i in range(50):
                doc = docs[(num_steps + i) % len(docs)]
                tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
                n = min(model.block_size, len(tokens) - 1)
                
                keys_cache = [torch.zeros(0, model.n_embd)]
                values_cache = [torch.zeros(0, model.n_embd)]
                q_keys_cache = [torch.zeros(0, model.head_dim)]
                
                losses = []
                for pos_id in range(n):
                    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                    token_tensor = torch.tensor([token_id], dtype=torch.long)
                    target_tensor = torch.tensor([target_id], dtype=torch.long)
                    
                    logits = model(token_tensor, pos_id, keys_cache, values_cache, q_keys_cache)
                    loss_t = criterion(logits.unsqueeze(0), target_tensor)
                    losses.append(loss_t)
                val_losses.append((sum(losses) / n).item())
        noisy_val_loss = sum(val_losses) / len(val_losses)
        print(f"Noisy Validation Loss: {noisy_val_loss:.4f}")
        
        # Switch back to clean for sample inference
        model.set_noisy_simulator(use_noisy=False)

    # 6. Inference
    temperature = 0.5
    print("\n--- Generating samples ---")
    inference_samples = []
    model.eval()
    with torch.no_grad():
        for sample_idx in range(5):
            keys_cache = [torch.zeros(0, model.n_embd)]
            values_cache = [torch.zeros(0, model.n_embd)]
            q_keys_cache = [torch.zeros(0, model.head_dim)]
            token_id = BOS
            sample = []
            for pos_id in range(model.block_size):
                token_tensor = torch.tensor([token_id], dtype=torch.long)
                logits = model(token_tensor, pos_id, keys_cache, values_cache, q_keys_cache)
                probs = torch.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                if token_id == BOS:
                    break
                sample.append(uchars[token_id])
            generated_name = "".join(sample)
            inference_samples.append(generated_name)
            print(f"  sample {sample_idx+1}: {generated_name}")

    return {
        "total_params": total_params,
        "quantum_params": quantum_params,
        "training_time": training_time,
        "losses": all_step_losses,
        "final_loss": all_step_losses[-1],
        "noisy_val_loss": noisy_val_loss,
        "inference_samples": inference_samples
    }

if __name__ == "__main__":
    train_and_evaluate(use_noisy_eval=True)
