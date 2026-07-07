"""
microgpt_quantum.py

A PyTorch port of Andrej Karpathy's microGPT where Head 0 is CLASSICAL and Heads 1,2,3 are QUANTUM (3 quantum heads total).
Each quantum head uses a separate Parameterized Quantum Circuit (VQC) via Qiskit Machine Learning.

========================================================================================
QUANTUM HEAD AND GRADIENT FLOW DESCRIPTION:
- Each Quantum Head (h=1,2,3) has its own:
  - Classical projection adapters (in/out projections)
  - Separate Parameterized Quantum Circuit (VQC)
- In forward pass, Query (Q) and Key (K) slices for quantum heads are classically projected,
  scaled to [-pi, pi] using tanh(x)*pi, and passed through VQC
- VQCs use 4-qubit Angle Embedding (RY rotations) + variational ansatz (RX+RY+CNOT ring) + Pauli-Z expectation measurements
- For backpropagation: input_gradients=True in EstimatorQNN enables parameter-shift rule for circuit weights and inputs
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
from config import Hyperparameters as hp

class RMSNorm(nn.Module):
    """Parameter-free RMSNorm exactly matching microGPT's definition."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)

class QuantumGPT(nn.Module):
    def __init__(self, vocab_size, use_noisy=False):
        super().__init__()
        torch.manual_seed(hp.seed)
        random.seed(hp.seed)
        
        self.n_embd = hp.embed_dim
        self.block_size = hp.seq_len
        self.n_head = hp.num_heads
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = hp.num_layers
        self.num_quantum_heads = hp.num_quantum_heads
        self.num_qubits = hp.num_qubits
        
        # Embedding Layers
        self.wte = nn.Embedding(vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.block_size, self.n_embd)
        
        # Attention Projection Weights
        self.attn_wq = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.attn_wk = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.attn_wv = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.attn_wo = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # MLP Layers
        self.mlp_fc1 = nn.Linear(self.n_embd, hp.ffn_dim, bias=False)
        self.mlp_fc2 = nn.Linear(hp.ffn_dim, self.n_embd, bias=False)
        
        # LM Head
        self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)
        
        # --- Classical Projection Adapters & VQCs for Quantum Heads ---
        self.q_in_projs = nn.ModuleList()
        self.q_out_projs = nn.ModuleList()
        self.k_in_projs = nn.ModuleList()
        self.k_out_projs = nn.ModuleList()
        self.vqcs = nn.ModuleList()
        
        for _ in range(self.num_quantum_heads):
            # Classical adapters
            self.q_in_projs.append(nn.Linear(self.head_dim, self.head_dim, bias=False))
            self.q_out_projs.append(nn.Linear(self.head_dim, self.head_dim, bias=False))
            self.k_in_projs.append(nn.Linear(self.head_dim, self.head_dim, bias=False))
            self.k_out_projs.append(nn.Linear(self.head_dim, self.head_dim, bias=False))
            
            # Build Parameterized Quantum Circuit (PQC) for this head
            inputs = ParameterVector("x", self.head_dim)
            weights = ParameterVector("w", 2 * self.head_dim)
            
            qc = QuantumCircuit(self.head_dim)
            # Angle Embedding Feature Map
            for i in range(self.head_dim):
                qc.ry(inputs[i], i)
            # Parameterized Ansatz Rotations
            for i in range(self.head_dim):
                qc.rx(weights[2*i], i)
                qc.ry(weights[2*i+1], i)
            # Entangling Ring of CNOTs
            for i in range(self.head_dim - 1):
                qc.cx(i, i+1)
            qc.cx(self.head_dim - 1, 0)
            
            # Observables (Z expectation value on all qubits)
            observables = []
            for q in range(self.head_dim):
                pauli_list = ["I"] * self.head_dim
                pauli_list[q] = "Z"
                pauli_str = "".join(reversed(pauli_list))
                observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
            
            # Estimator & QNN Setup
            estimator = self._get_estimator(use_noisy)
            qnn = EstimatorQNN(
                circuit=qc,
                input_params=inputs,
                weight_params=weights,
                observables=observables,
                estimator=estimator,
                input_gradients=True
            )
            self.vqcs.append(TorchConnector(qnn))
        
        # RMSNorm blocks
        self.rmsnorm1 = RMSNorm(self.n_embd)
        self.rmsnorm2 = RMSNorm(self.n_embd)
        self.rmsnorm3 = RMSNorm(self.n_embd)
        
        # Gaussian Initialization (std=0.08) matching microGPT
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.08)

    def _get_estimator(self, use_noisy, shots=hp.shots, depol_error=hp.depolarizing_error):
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

    def set_noisy_simulator(self, use_noisy, shots=hp.shots, depol_error=hp.depolarizing_error):
        """Swaps Qiskit Estimators for ALL VQCs to noisy AerSimulator configuration."""
        for vqc_idx in range(self.num_quantum_heads):
            # Save old weights for continuity
            old_weights = self.vqcs[vqc_idx].weight.data.clone()
            
            # Rebuild circuit & QNN for each quantum head
            inputs = ParameterVector("x", self.head_dim)
            weights = ParameterVector("w", 2 * self.head_dim)
            
            qc = QuantumCircuit(self.head_dim)
            for i in range(self.head_dim):
                qc.ry(inputs[i], i)
            for i in range(self.head_dim):
                qc.rx(weights[2*i], i)
                qc.ry(weights[2*i+1], i)
            for i in range(self.head_dim - 1):
                qc.cx(i, i+1)
            qc.cx(self.head_dim - 1, 0)
            
            observables = []
            for q in range(self.head_dim):
                pauli_list = ["I"] * self.head_dim
                pauli_list[q] = "Z"
                pauli_str = "".join(reversed(pauli_list))
                observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
            
            estimator = self._get_estimator(use_noisy, shots, depol_error)
            qnn = EstimatorQNN(
                circuit=qc,
                input_params=inputs,
                weight_params=weights,
                observables=observables,
                estimator=estimator,
                input_gradients=True
            )
            self.vqcs[vqc_idx] = TorchConnector(qnn, initial_weights=old_weights)

    def forward(self, token_id, pos_id, keys_cache, values_cache, q_keys_caches):
        # Capture the compute device once (GPU if CUDA available, else CPU).
        # Qiskit VQCs only run on CPU; we bridge back to `compute_device` after each call.
        compute_device = token_id.device

        # Embeddings
        tok_emb = self.wte(token_id).squeeze(0)
        pos_emb = self.wpe(torch.tensor([pos_id], device=compute_device)).squeeze(0)
        x = tok_emb + pos_emb
        x = self.rmsnorm1(x)
        
        # Layer 0 Attention block
        x_residual = x
        x = self.rmsnorm2(x)
        
        q = self.attn_wq(x)
        k = self.attn_wk(x)
        v = self.attn_wv(x)
        
        # Update KV cache
        keys_cache[0] = torch.cat([keys_cache[0], k.unsqueeze(0)], dim=0)
        values_cache[0] = torch.cat([values_cache[0], v.unsqueeze(0)], dim=0)
        
        head_outputs = []
        
        for h in range(self.n_head):
            hs = h * self.head_dim
            q_h = q[hs : hs+self.head_dim]
            k_h = keys_cache[0][pos_id, hs : hs+self.head_dim]
            v_h = values_cache[0][:, hs : hs+self.head_dim]
            
            if h == 0:
                # --- Classical Head (fully on GPU) ---
                k_h_cache = keys_cache[0][:, hs : hs+self.head_dim]
                attn_logits = torch.matmul(q_h.unsqueeze(0), k_h_cache.transpose(0, 1)).squeeze(0) / math.sqrt(self.head_dim)
                attn_weights = torch.softmax(attn_logits, dim=-1)
                head_out = torch.matmul(attn_weights.unsqueeze(0), v_h).squeeze(0)
            else:
                # --- Quantum Head (VQC runs on CPU; bridge in/out) ---
                q_idx = h - 1
                
                # 1. Project Query, scale, move to CPU, run VQC, move back to compute_device
                q_h_proj = self.q_in_projs[q_idx](q_h.unsqueeze(0))
                q_h_scaled = torch.tanh(q_h_proj) * torch.pi
                q_h_vqc = self.vqcs[q_idx](q_h_scaled.cpu()).to(compute_device)
                q_h_prime = self.q_out_projs[q_idx](q_h_vqc).squeeze(0)
                
                # 2. Project Key, scale, move to CPU, run VQC, move back to compute_device
                k_h_proj = self.k_in_projs[q_idx](k_h.unsqueeze(0))
                k_h_scaled = torch.tanh(k_h_proj) * torch.pi
                k_h_vqc = self.vqcs[q_idx](k_h_scaled.cpu()).to(compute_device)
                k_h_prime = self.k_out_projs[q_idx](k_h_vqc)
                
                # 3. Update quantum key cache for this head
                q_keys_caches[q_idx] = torch.cat([q_keys_caches[q_idx], k_h_prime], dim=0)
                
                # 4. Dot-product attention scores
                attn_logits = torch.matmul(q_h_prime.unsqueeze(0), q_keys_caches[q_idx].transpose(0, 1)).squeeze(0) / math.sqrt(self.head_dim)
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
    print("(Head 0: Classical, Heads 1-3: Quantum)")
    print("=" * 60)
    
    torch.manual_seed(hp.seed)
    random.seed(hp.seed)
    
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

    # Move all classical PyTorch parameters (embeddings, projections, MLP, LM head)
    # to GPU. TorchConnector weights are registered as nn.Parameters but the VQC
    # execution always runs on CPU — the device bridge in forward() handles that.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"[Device] Using GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_params = sum(vqc.weight.numel() for vqc in model.vqcs)
    print(f"Total Params: {total_params} | Quantum Params: {quantum_params}")

    # 4. Optimizer and training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, betas=(0.85, 0.99), eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    all_step_losses = []
    
    start_time = time.perf_counter()
    
    # Run training
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(model.block_size, len(tokens) - 1)

        # Initialize KV Cache and QUANTUM KEY CACHES on the compute device
        keys_cache = [torch.zeros(0, model.n_embd, device=device)]
        values_cache = [torch.zeros(0, model.n_embd, device=device)]
        q_keys_caches = [torch.zeros(0, model.head_dim, device=device) for _ in range(model.num_quantum_heads)]
        
        optimizer.zero_grad()
        
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            token_tensor = torch.tensor([token_id], dtype=torch.long, device=device)
            target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
            
            logits = model(token_tensor, pos_id, keys_cache, values_cache, q_keys_caches)
            loss_t = criterion(logits.unsqueeze(0), target_tensor)
            losses.append(loss_t)
            
        loss = sum(losses) / n
        all_step_losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        
        # Adam optimizer update with linear LR decay
        lr_t = hp.learning_rate * (1 - step / num_steps)
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
        print("\nSwitching to noisy AerSimulator for evaluation...")
        model.set_noisy_simulator(use_noisy=True)
        model.eval()
        
        # Evaluate on next 50 validation documents
        val_losses = []
        with torch.no_grad():
            for i in range(50):
                doc = docs[(num_steps + i) % len(docs)]
                tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
                n = min(model.block_size, len(tokens) - 1)
                
                keys_cache = [torch.zeros(0, model.n_embd, device=device)]
                values_cache = [torch.zeros(0, model.n_embd, device=device)]
                q_keys_caches = [torch.zeros(0, model.head_dim, device=device) for _ in range(model.num_quantum_heads)]
                
                losses = []
                for pos_id in range(n):
                    token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                    token_tensor = torch.tensor([token_id], dtype=torch.long, device=device)
                    target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
                    
                    logits = model(token_tensor, pos_id, keys_cache, values_cache, q_keys_caches)
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
            keys_cache = [torch.zeros(0, model.n_embd, device=device)]
            values_cache = [torch.zeros(0, model.n_embd, device=device)]
            q_keys_caches = [torch.zeros(0, model.head_dim, device=device) for _ in range(model.num_quantum_heads)]
            token_id = BOS
            sample = []
            for pos_id in range(model.block_size):
                token_tensor = torch.tensor([token_id], dtype=torch.long, device=device)
                logits = model(token_tensor, pos_id, keys_cache, values_cache, q_keys_caches)
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
    train_and_evaluate(use_noisy_eval=False)
