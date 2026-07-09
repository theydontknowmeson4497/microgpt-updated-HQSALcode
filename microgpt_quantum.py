"""
microgpt_quantum.py

A PyTorch port of Andrej Karpathy's microGPT where Head 0 is CLASSICAL and Heads 1,2,3 are QUANTUM.
Uses word-level tokenization: each unique word in input.txt becomes its own token ID,
giving a vocabulary of ~32k tokens so all words are represented as model parameters.

========================================================================================
QUANTUM HEAD AND GRADIENT FLOW DESCRIPTION:
- Each Quantum Head (h=1,2,3) has its own:
  - Classical projection adapters: head_dim -> num_qubits -> VQC -> num_qubits -> head_dim
  - Separate Parameterized Quantum Circuit (VQC) with fixed num_qubits (tractable regardless of embed_dim)
- In forward pass, Q and K for quantum heads are projected down to num_qubits,
  scaled to [-pi, pi] using tanh(x)*pi, and passed through VQC
- VQCs use Angle Embedding (RY) + variational ansatz (RX+RY+CNOT ring) + Pauli-Z expectation
- input_gradients=True in EstimatorQNN enables parameter-shift rule for gradients
- Multi-layer: each transformer layer has its own attention/MLP weights and KV cache
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
    """Parameter-free RMSNorm matching microGPT's definition."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


# Global cache for built VQCs to avoid re-building them unnecessarily
_VQC_CACHE = {}


def _build_vqc(num_qubits, q_depth, use_noisy, shots, depol_error):
    """Builds one VQC (PQC + QNN + TorchConnector) with fixed num_qubits, with caching and optimizations."""
    cache_key = (num_qubits, q_depth, use_noisy, shots, depol_error)
    if cache_key in _VQC_CACHE:
        return _VQC_CACHE[cache_key]

    inputs  = ParameterVector("x", num_qubits)
    weights = ParameterVector("w", 2 * num_qubits * q_depth)

    qc = QuantumCircuit(num_qubits)
    # Angle embedding feature map
    for i in range(num_qubits):
        qc.ry(inputs[i], i)
    # Variational ansatz (depth layers of RX+RY rotations + CNOT ring)
    param_idx = 0
    for _ in range(q_depth):
        for i in range(num_qubits):
            qc.rx(weights[param_idx],     i)
            qc.ry(weights[param_idx + 1], i)
            param_idx += 2
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        if num_qubits > 2:
            qc.cx(num_qubits - 1, 0)

    # Pauli-Z observable on each qubit
    observables = []
    for q in range(num_qubits):
        pauli_list = ["I"] * num_qubits
        pauli_list[q] = "Z"
        pauli_str = "".join(reversed(pauli_list))
        observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    # Estimator primitive - use fastest possible settings
    if not use_noisy:
        estimator = StatevectorEstimator()
    else:
        from qiskit_aer.primitives import EstimatorV2
        from qiskit_aer.noise import NoiseModel, depolarizing_error as dep_err
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(dep_err(depol_error, 1),     ["rx", "ry"])
        noise_model.add_all_qubit_quantum_error(dep_err(depol_error * 2, 2), ["cx"])
        estimator = EstimatorV2(options={
            "run_options":     {"shots": shots},
            "backend_options": {"noise_model": noise_model}
        })

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        observables=observables,
        estimator=estimator,
        input_gradients=True
    )
    vqc = TorchConnector(qnn)
    _VQC_CACHE[cache_key] = vqc
    return vqc


# Optimized quantum connector with minimal data transfers and no-copy where possible
@torch.jit.ignore
def _run_quantum_connector(connector, x):
    """Run quantum connector with minimal CPU-GPU data transfer overhead."""
    # Move to CPU - avoid unnecessary gradient tracking on transfer for inputs
    if x.requires_grad:
        # For training: we need gradients, so move to CPU normally
        x_cpu = x.to("cpu")
        out_cpu = connector(x_cpu)
        return out_cpu.to(x.device)
    else:
        # For inference: no gradients, faster transfer
        x_cpu = x.to("cpu", copy=False)
        out_cpu = connector(x_cpu)
        return out_cpu.to(x.device, copy=False)


class QuantumGPT(nn.Module):
    def __init__(self, vocab_size, use_noisy=False, device=None, compile_classical=True):
        super().__init__()
        torch.manual_seed(hp.seed)
        random.seed(hp.seed)

        self.classical_device = torch.device(device or hp.device)
        self.quantum_device = torch.device("cpu")

        self.n_embd           = hp.embed_dim
        self.block_size       = hp.seq_len
        self.n_head           = hp.num_heads
        self.head_dim         = self.n_embd // self.n_head
        self.n_layer          = hp.num_layers
        self.num_quantum_heads = hp.num_quantum_heads
        self.num_qubits       = hp.num_qubits   # fixed; head_dim projected down to this

        # Token + position embeddings
        # wte has vocab_size rows → every unique word gets its own embedding vector
        self.wte = nn.Embedding(vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.block_size, self.n_embd)

        # Per-layer attention projections
        self.attn_wq = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])
        self.attn_wk = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])
        self.attn_wv = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])
        self.attn_wo = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])

        # Per-layer MLP
        self.mlp_fc1 = nn.ModuleList([nn.Linear(self.n_embd, hp.ffn_dim, bias=False) for _ in range(self.n_layer)])
        self.mlp_fc2 = nn.ModuleList([nn.Linear(hp.ffn_dim, self.n_embd, bias=False) for _ in range(self.n_layer)])

        # Per-layer RMSNorm (pre-attn and pre-mlp) + one for embeddings
        self.rmsnorm_attn = nn.ModuleList([RMSNorm(self.n_embd) for _ in range(self.n_layer)])
        self.rmsnorm_mlp  = nn.ModuleList([RMSNorm(self.n_embd) for _ in range(self.n_layer)])
        self.rmsnorm_emb  = RMSNorm(self.n_embd)

        # LM head
        self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)

        # Quantum heads: classical adapters (head_dim <-> num_qubits) + VQCs
        # Adapters are shared across layers; VQCs are one per quantum head
        self.q_in_projs  = nn.ModuleList()
        self.q_out_projs = nn.ModuleList()
        self.k_in_projs  = nn.ModuleList()
        self.k_out_projs = nn.ModuleList()
        self.vqcs        = nn.ModuleList()

        for _ in range(self.num_quantum_heads):
            self.q_in_projs.append( nn.Linear(self.head_dim, self.num_qubits, bias=False))
            self.q_out_projs.append(nn.Linear(self.num_qubits, self.head_dim, bias=False))
            self.k_in_projs.append( nn.Linear(self.head_dim, self.num_qubits, bias=False))
            self.k_out_projs.append(nn.Linear(self.num_qubits, self.head_dim, bias=False))
            self.vqcs.append(_build_vqc(self.num_qubits, hp.q_depth,
                                         use_noisy, hp.shots, hp.depolarizing_error))

        self.apply(self._init_weights)
        self.to(self.classical_device)
        for vqc in self.vqcs:
            vqc.to(self.quantum_device)
            
        # Optional: compile classical layers for speed
        if compile_classical and torch.cuda.is_available():
            try:
                # Compile linear layers and RMSNorm for faster execution
                # Skip compiling VQC-adjacent layers since they interface with CPU
                for name, module in self.named_modules():
                    if isinstance(module, (nn.Linear, RMSNorm)) and 'q_' not in name and 'k_' not in name:
                        try:
                            setattr(self, name, torch.compile(module, mode='max-autotune'))
                        except Exception:
                            pass
            except Exception:
                pass

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        for vqc in getattr(module, "vqcs", []):
            vqc.to(torch.device("cpu"))
        return module

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.08)

    def set_noisy_simulator(self, use_noisy,
                             shots=hp.shots, depol_error=hp.depolarizing_error):
        """Swap all VQCs to noisy (or back to clean) estimator, preserving weights."""
        for i in range(self.num_quantum_heads):
            old_w = self.vqcs[i].weight.data.clone()
            new_vqc = _build_vqc(self.num_qubits, hp.q_depth,
                                  use_noisy, shots, depol_error)
            # Re-wrap with old weights
            from qiskit_machine_learning.connectors import TorchConnector as TC
            self.vqcs[i] = TC(new_vqc._module, initial_weights=old_w)  # type: ignore
            self.vqcs[i].to(self.quantum_device)

    def forward(self, token_id, pos_id, keys_cache, values_cache, q_keys_caches):
        """
        token_id    : LongTensor [1]
        pos_id      : int  (position in the current sequence)
        keys_cache  : list[Tensor]  length n_layer, each [pos, n_embd]
        values_cache: list[Tensor]  length n_layer, each [pos, n_embd]
        q_keys_caches: list[Tensor] length n_layer * num_quantum_heads,
                       each [pos, head_dim]  (indexed as li*num_quantum_heads + q_idx)
        """
        # Embeddings - use integer directly for position to avoid unnecessary tensor creation
        x = self.wte(token_id).squeeze(0) + self.wpe.weight[pos_id]
        x = self.rmsnorm_emb(x)

        for li in range(self.n_layer):
            # ---- Attention ----
            x_res = x
            x = self.rmsnorm_attn[li](x)

            q = self.attn_wq[li](x)
            k = self.attn_wk[li](x)
            v = self.attn_wv[li](x)

            keys_cache[li]   = torch.cat([keys_cache[li],   k.unsqueeze(0)], dim=0)
            values_cache[li] = torch.cat([values_cache[li], v.unsqueeze(0)], dim=0)

            head_outputs = []
            for h in range(self.n_head):
                hs  = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                v_h = values_cache[li][:, hs : hs + self.head_dim]

                if h == 0:
                    # Classical head
                    k_cache     = keys_cache[li][:, hs : hs + self.head_dim]
                    scores      = torch.matmul(q_h.unsqueeze(0), k_cache.T).squeeze(0) \
                                  / math.sqrt(self.head_dim)
                    attn_w      = torch.softmax(scores, dim=-1)
                    head_out    = torch.matmul(attn_w.unsqueeze(0), v_h).squeeze(0)
                else:
                    # Quantum head
                    q_idx       = h - 1
                    cache_idx   = li * self.num_quantum_heads + q_idx

                    # Query: head_dim -> num_qubits -> VQC -> head_dim
                    q_proj      = self.q_in_projs[q_idx](q_h.unsqueeze(0))
                    q_vqc_out   = _run_quantum_connector(self.vqcs[q_idx], torch.tanh(q_proj) * math.pi)
                    q_prime     = self.q_out_projs[q_idx](q_vqc_out).squeeze(0)

                    # Key at current pos: head_dim -> num_qubits -> VQC -> head_dim
                    k_cur       = keys_cache[li][pos_id, hs : hs + self.head_dim]
                    k_proj      = self.k_in_projs[q_idx](k_cur.unsqueeze(0))
                    k_vqc_out   = _run_quantum_connector(self.vqcs[q_idx], torch.tanh(k_proj) * math.pi)
                    k_prime     = self.k_out_projs[q_idx](k_vqc_out)   # [1, head_dim]

                    q_keys_caches[cache_idx] = torch.cat(
                        [q_keys_caches[cache_idx], k_prime], dim=0)

                    scores      = torch.matmul(q_prime.unsqueeze(0),
                                               q_keys_caches[cache_idx].T).squeeze(0) \
                                  / math.sqrt(self.head_dim)
                    attn_w      = torch.softmax(scores, dim=-1)
                    head_out    = torch.matmul(attn_w.unsqueeze(0), v_h).squeeze(0)

                head_outputs.append(head_out)

            x = self.attn_wo[li](torch.cat(head_outputs, dim=-1)) + x_res

            # ---- MLP ----
            x_res = x
            x = self.rmsnorm_mlp[li](x)
            x = torch.relu(self.mlp_fc1[li](x))
            x = self.mlp_fc2[li](x) + x_res

        return self.lm_head(x)


def train_and_evaluate(use_noisy_eval=False):
    print("=" * 70)
    print("STARTING HYBRID QUANTUM MICROGPT TRAINING")
    print(f"(Head 0: Classical, Heads 1-{hp.num_heads-1}: Quantum)")
    print("=" * 70)

    torch.manual_seed(hp.seed)
    random.seed(hp.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # Enable cuDNN benchmark mode for faster convs (though we don't use convs here, it can't hurt)
        torch.backends.cudnn.benchmark = True

    device = torch.device(hp.device)

    # 1. Load dataset
    if not os.path.exists('input.txt'):
        import urllib.request
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt',
            'input.txt')
    docs = [line.strip() for line in open('input.txt') if line.strip()]
    print(f"total words loaded: {len(docs)}")

    # 2. Word-level tokenizer
    # Every unique word gets its own token ID → vocab covers all 32k words
    unique_words = sorted(set(docs))
    word_to_idx  = {w: i for i, w in enumerate(unique_words)}
    BOS          = len(unique_words)          # boundary / end-of-sequence token
    vocab_size   = len(unique_words) + 1
    print(f"vocab size (unique words + BOS): {vocab_size}")

    # Clear old VQC cache to pick up new gradient setting
    global _VQC_CACHE
    _VQC_CACHE = {}

    # 3. Build model
    model = QuantumGPT(vocab_size=vocab_size, use_noisy=False, device=device, compile_classical=True)
    total_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_params = sum(vqc.weight.numel() for vqc in model.vqcs)
    print(f"Total Params: {total_params:,} | Quantum Params: {quantum_params}")

    # 4. Optimizer (fused Adam if available)
    optimizer_kwargs = {
        "lr": hp.learning_rate,
        "betas": (0.85, 0.99),
        "eps": 1e-8
    }
    # Try to use fused Adam if available (faster on GPU)
    if torch.cuda.is_available():
        try:
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs, fused=True)
        except TypeError:
            # Fused Adam not available, fall back to regular
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    criterion = nn.CrossEntropyLoss()

    all_step_losses = []
    start_time      = time.perf_counter()

    # Pre-allocate cache tensors (optional small optimization)
    empty_embd = torch.empty(0, model.n_embd, device=device)
    empty_head = torch.empty(0, model.head_dim, device=device)

    # 5. Training — capped at max_train_steps to stay within 16 GB RAM
    num_steps = min(len(docs), hp.max_train_steps)
    print(f"Training for {num_steps} steps (vocab covers all {len(docs)} words)")
    for step in range(num_steps):
        doc    = docs[step]
        tokens = [BOS, word_to_idx[doc], BOS]   # BOS <word> BOS
        n      = min(model.block_size, len(tokens) - 1)

        # KV cache: one entry per layer; quantum key cache: one per layer×quantum_head
        keys_cache    = [empty_embd.clone() for _ in range(model.n_layer)]
        values_cache  = [empty_embd.clone() for _ in range(model.n_layer)]
        q_keys_caches = [empty_head.clone() for _ in range(model.n_layer * model.num_quantum_heads)]

        optimizer.zero_grad(set_to_none=True)  # Use set_to_none for faster zero_grad

        losses = []
        for pos_id in range(n):
            token_id  = tokens[pos_id]
            target_id = tokens[pos_id + 1]
            logits    = model(torch.tensor([token_id], dtype=torch.long, device=device),
                              pos_id, keys_cache, values_cache, q_keys_caches)
            losses.append(criterion(logits.unsqueeze(0),
                                    torch.tensor([target_id], dtype=torch.long, device=device)))

        loss = sum(losses) / n
        all_step_losses.append(loss.item())

        loss.backward()

        # Linear LR decay
        for pg in optimizer.param_groups:
            pg['lr'] = hp.learning_rate * (1 - step / num_steps)
        optimizer.step()

        if step == 0 or (step + 1) % 1000 == 0:
                print(f"Step {step+1:5d}/{num_steps} | Loss: {loss.item():.4f}")

    training_time = time.perf_counter() - start_time
    print(f"Finished training in {training_time:.2f} seconds.")

    # 6. Optional noisy evaluation
    noisy_val_loss = "N/A"
    if use_noisy_eval:
        print("\nSwitching to noisy AerSimulator for evaluation...")
        model.set_noisy_simulator(use_noisy=True)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(50):
                doc    = docs[(num_steps + i) % len(docs)]
                tokens = [BOS, word_to_idx[doc], BOS]
                n      = min(model.block_size, len(tokens) - 1)
                keys_cache    = [torch.zeros(0, model.n_embd, device=device) for _ in range(model.n_layer)]
                values_cache  = [torch.zeros(0, model.n_embd, device=device) for _ in range(model.n_layer)]
                q_keys_caches = [torch.zeros(0, model.head_dim, device=device)
                                  for _ in range(model.n_layer * model.num_quantum_heads)]
                v_losses = []
                for pos_id in range(n):
                    token_id  = tokens[pos_id]
                    target_id = tokens[pos_id + 1]
                    logits    = model(torch.tensor([token_id], dtype=torch.long, device=device),
                                      pos_id, keys_cache, values_cache, q_keys_caches)
                    v_losses.append(criterion(logits.unsqueeze(0),
                                              torch.tensor([target_id], dtype=torch.long, device=device)))
                val_losses.append((sum(v_losses) / n).item())
        noisy_val_loss = sum(val_losses) / len(val_losses)
        print(f"Noisy Validation Loss: {noisy_val_loss:.4f}")
        model.set_noisy_simulator(use_noisy=False)

    # 7. Inference — generate word sequences
    temperature = 0.5
    print("\n--- Generating samples ---")
    inference_samples = []
    model.eval()
    with torch.no_grad():
        for sample_idx in range(5):
            keys_cache    = [torch.zeros(0, model.n_embd, device=device) for _ in range(model.n_layer)]
            values_cache  = [torch.zeros(0, model.n_embd, device=device) for _ in range(model.n_layer)]
            q_keys_caches = [torch.zeros(0, model.head_dim, device=device)
                              for _ in range(model.n_layer * model.num_quantum_heads)]
            token_id = BOS
            sample   = []
            for pos_id in range(model.block_size):
                logits   = model(torch.tensor([token_id], dtype=torch.long, device=device),
                                  pos_id, keys_cache, values_cache, q_keys_caches)
                probs    = torch.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                if token_id == BOS:
                    break
                sample.append(unique_words[token_id])
            generated = " ".join(sample)
            inference_samples.append(generated)
            print(f"  sample {sample_idx+1}: {generated}")

    return {
        "total_params":      total_params,
        "quantum_params":    quantum_params,
        "training_time":     training_time,
        "losses":            all_step_losses,
        "final_loss":        all_step_losses[-1],
        "noisy_val_loss":    noisy_val_loss,
        "inference_samples": inference_samples,
    }


if __name__ == "__main__":
    train_and_evaluate(use_noisy_eval=False)
