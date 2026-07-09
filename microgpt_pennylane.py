"""
microgpt_pennylane.py
A PyTorch + PennyLane hybrid quantum microGPT:
Head 0 = Classical, Heads 1-3 = Quantum (with lightning.gpu support for GPU acceleration!)
Uses word-level tokenization, matches the exact architecture of your original Qiskit model.
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
import pennylane as qml
from config import Hyperparameters as hp


# RMSNorm (same as before!)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


def _build_pennylane_vqc(num_qubits, q_depth, device_name="lightning.qubit"):
    """Build a PennyLane quantum node for our VQC"""
    # Try to use lightning.gpu if available! Fall back to lightning.qubit if not
    try:
        if device_name == "lightning.gpu" and torch.cuda.is_available():
            dev = qml.device("lightning.gpu", wires=num_qubits)
        else:
            dev = qml.device(device_name, wires=num_qubits)
    except Exception:
        dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def vqc(inputs, weights):
        # Angle embedding (like original Qiskit model)
        for i in range(num_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational ansatz (depth q_depth layers)
        for layer_idx in range(q_depth):
            # RX + RY rotations per qubit
            for i in range(num_qubits):
                qml.RX(weights[layer_idx, i, 0], wires=i)
                qml.RY(weights[layer_idx, i, 1], wires=i)
            # CNOT ring (like original)
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if num_qubits > 2:
                qml.CNOT(wires=[num_qubits - 1, 0])

        # Return Pauli-Z expectation for each qubit (matches original observables)
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(num_qubits))

    return vqc


class QuantumGPT(nn.Module):
    def __init__(self, vocab_size, use_gpu=True, device=None):
        super().__init__()
        torch.manual_seed(hp.seed)
        random.seed(hp.seed)

        self.classical_device = torch.device(device or hp.device)
        self.n_embd = hp.embed_dim
        self.block_size = hp.seq_len
        self.n_head = hp.num_heads
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = hp.num_layers
        self.num_quantum_heads = hp.num_quantum_heads
        self.num_qubits = hp.num_qubits
        self.q_depth = hp.q_depth

        # Token + position embeddings
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

        # Per-layer RMSNorm
        self.rmsnorm_attn = nn.ModuleList([RMSNorm(self.n_embd) for _ in range(self.n_layer)])
        self.rmsnorm_mlp = nn.ModuleList([RMSNorm(self.n_embd) for _ in range(self.n_layer)])
        self.rmsnorm_emb = RMSNorm(self.n_embd)

        # LM head
        self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)

        # Quantum heads: classical adapters + PennyLane TorchLayers
        self.q_in_projs = nn.ModuleList()
        self.q_out_projs = nn.ModuleList()
        self.k_in_projs = nn.ModuleList()
        self.k_out_projs = nn.ModuleList()
        self.vqc_layers = nn.ModuleList()

        # Initialize weights for VQC
        weight_shapes = {"weights": (self.q_depth, self.num_qubits, 2)}
        device_name = "lightning.gpu" if use_gpu and torch.cuda.is_available() else "lightning.qubit"

        for _ in range(self.num_quantum_heads):
            self.q_in_projs.append(nn.Linear(self.head_dim, self.num_qubits, bias=False))
            self.q_out_projs.append(nn.Linear(self.num_qubits, self.head_dim, bias=False))
            self.k_in_projs.append(nn.Linear(self.head_dim, self.num_qubits, bias=False))
            self.k_out_projs.append(nn.Linear(self.num_qubits, self.head_dim, bias=False))

            qnode = _build_pennylane_vqc(self.num_qubits, self.q_depth, device_name=device_name)
            self.vqc_layers.append(qml.qnn.TorchLayer(qnode, weight_shapes))

        self.apply(self._init_weights)
        self.to(self.classical_device)

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        return module

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.08)

    def forward(self, token_id, pos_id, keys_cache, values_cache, q_keys_caches):
        # Embeddings
        x = self.wte(token_id).squeeze(0) + self.wpe.weight[pos_id]
        x = self.rmsnorm_emb(x)

        for li in range(self.n_layer):
            # Attention
            x_res = x
            x = self.rmsnorm_attn[li](x)

            q = self.attn_wq[li](x)
            k = self.attn_wk[li](x)
            v = self.attn_wv[li](x)

            keys_cache[li] = torch.cat([keys_cache[li], k.unsqueeze(0)], dim=0)
            values_cache[li] = torch.cat([values_cache[li], v.unsqueeze(0)], dim=0)

            head_outputs = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs + self.head_dim]
                v_h = values_cache[li][:, hs:hs + self.head_dim]

                if h == 0:
                    # Classical head (same as original)
                    k_cache = keys_cache[li][:, hs:hs + self.head_dim]
                    scores = torch.matmul(q_h.unsqueeze(0), k_cache.T).squeeze(0) / math.sqrt(self.head_dim)
                    attn_w = torch.softmax(scores, dim=-1)
                    head_out = torch.matmul(attn_w.unsqueeze(0), v_h).squeeze(0)
                else:
                    # Quantum head (PennyLane!)
                    q_idx = h - 1
                    cache_idx = li * self.num_quantum_heads + q_idx

                    # Query path
                    q_proj = self.q_in_projs[q_idx](q_h.unsqueeze(0))
                    q_inputs = torch.tanh(q_proj) * math.pi
                    q_vqc_out = self.vqc_layers[q_idx](q_inputs[0]).flatten().unsqueeze(0)
                    q_prime = self.q_out_projs[q_idx](q_vqc_out).squeeze(0)

                    # Key path (current position only)
                    k_cur = keys_cache[li][pos_id, hs:hs + self.head_dim]
                    k_proj = self.k_in_projs[q_idx](k_cur.unsqueeze(0))
                    k_inputs = torch.tanh(k_proj) * math.pi
                    k_vqc_out = self.vqc_layers[q_idx](k_inputs[0]).flatten().unsqueeze(0)
                    k_prime = self.k_out_projs[q_idx](k_vqc_out)

                    q_keys_caches[cache_idx] = torch.cat([q_keys_caches[cache_idx], k_prime], dim=0)

                    scores = torch.matmul(q_prime.unsqueeze(0), q_keys_caches[cache_idx].T).squeeze(0) / math.sqrt(self.head_dim)
                    attn_w = torch.softmax(scores, dim=-1)
                    head_out = torch.matmul(attn_w.unsqueeze(0), v_h).squeeze(0)

                head_outputs.append(head_out)

            x = self.attn_wo[li](torch.cat(head_outputs, dim=-1)) + x_res

            # MLP
            x_res = x
            x = self.rmsnorm_mlp[li](x)
            x = torch.relu(self.mlp_fc1[li](x))
            x = self.mlp_fc2[li](x) + x_res

        return self.lm_head(x)


def train_and_evaluate(use_noisy_eval=False, use_gpu=True):
    print("=" * 70)
    print("STARTING HYBRID QUANTUM microGPT TRAINING (PENNYLANE)")
    print(f"(Head 0: Classical, Heads 1-{hp.num_heads-1}: Quantum)")
    if use_gpu and torch.cuda.is_available():
        print("Using lightning.gpu backend for quantum simulation!")
    print("=" * 70)

    torch.manual_seed(hp.seed)
    random.seed(hp.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    device = torch.device(hp.device)

    # Load dataset
    if not os.path.exists('input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, 'input.txt')
    docs = [line.strip() for line in open('input.txt') if line.strip()]
    print(f"total words loaded: {len(docs)}")

    # Word-level tokenizer
    unique_words = sorted(set(docs))
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    BOS = len(unique_words)
    vocab_size = len(unique_words) + 1
    print(f"vocab size (unique words + BOS): {vocab_size}")

    # Build model
    model = QuantumGPT(vocab_size=vocab_size, use_gpu=use_gpu, device=device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_params = sum(p.numel() for vqc in model.vqc_layers for p in vqc.parameters())
    print(f"Total Params: {total_params:,} | Quantum Params: {quantum_params}")

    # Optimizer
    optimizer_kwargs = {
        "lr": hp.learning_rate,
        "betas": (0.85, 0.99),
        "eps": 1e-8
    }
    if torch.cuda.is_available():
        try:
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs, fused=True)
        except TypeError:
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    criterion = nn.CrossEntropyLoss()
    all_step_losses = []
    start_time = time.perf_counter()
    num_steps = min(len(docs), hp.max_train_steps)
    print(f"Training for {num_steps} steps")

    # Pre-allocate empty caches
    empty_embd = torch.empty(0, model.n_embd, device=device)
    empty_head = torch.empty(0, model.head_dim, device=device)

    for step in range(num_steps):
        doc = docs[step]
        tokens = [BOS, word_to_idx[doc], BOS]
        n = min(model.block_size, len(tokens) - 1)

        keys_cache = [empty_embd.clone() for _ in range(model.n_layer)]
        values_cache = [empty_embd.clone() for _ in range(model.n_layer)]
        q_keys_caches = [empty_head.clone() for _ in range(model.n_layer * model.num_quantum_heads)]

        optimizer.zero_grad(set_to_none=True)
        losses = []

        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]
            logits = model(torch.tensor([token_id], dtype=torch.long, device=device),
                           pos_id, keys_cache, values_cache, q_keys_caches)
            losses.append(criterion(logits.unsqueeze(0),
                                     torch.tensor([target_id], dtype=torch.long, device=device)))

        loss = sum(losses) / n
        all_step_losses.append(loss.item())
        loss.backward()

        for pg in optimizer.param_groups:
            pg['lr'] = hp.learning_rate * (1 - step / num_steps)
        optimizer.step()

        if step == 0 or (step + 1) % 1000 == 0:
            print(f"Step {step+1:5d}/{num_steps} | Loss: {loss.item():.4f}")

    training_time = time.perf_counter() - start_time
    print(f"Finished training in {training_time:.2f} seconds.")

    # Optional noisy eval (not applicable for PennyLane lightning, skipped)
    noisy_val_loss = "N/A (PennyLane lightning)"
    if use_noisy_eval:
        print("\nNoisy evaluation not applicable for PennyLane lightning backends (use simulator with noise model if needed).")

    # Inference
    temperature = 0.5
    print("\n--- Generating samples ---")
    inference_samples = []
    model.eval()

    with torch.no_grad():
        for _ in range(5):
            keys_cache = [empty_embd.clone() for _ in range(model.n_layer)]
            values_cache = [empty_embd.clone() for _ in range(model.n_layer)]
            q_keys_caches = [empty_head.clone() for _ in range(model.n_layer * model.num_quantum_heads)]
            token_id = BOS
            sample = []

            for pos_id in range(model.block_size):
                logits = model(torch.tensor([token_id], dtype=torch.long, device=device),
                               pos_id, keys_cache, values_cache, q_keys_caches)
                probs = torch.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                if token_id == BOS:
                    break
                sample.append(unique_words[token_id])
            generated = " ".join(sample)
            inference_samples.append(generated)
            print(f"  sample: {generated}")

    return {
        "total_params": total_params,
        "quantum_params": quantum_params,
        "training_time": training_time,
        "losses": all_step_losses,
        "final_loss": all_step_losses[-1],
        "noisy_val_loss": noisy_val_loss,
        "inference_samples": inference_samples,
    }


if __name__ == "__main__":
    train_and_evaluate(use_noisy_eval=False, use_gpu=True)

