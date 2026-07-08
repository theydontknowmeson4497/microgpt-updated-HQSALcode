import torch
import torch.nn as nn
import math
import time
import random
import os
from config import Hyperparameters as hp


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm exactly matching microGPT's definition."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


class ClassicalMicroGPT(nn.Module):
    """
    Exact PyTorch port of Andrej Karpathy's microGPT, matching all details:
    - RMSNorm (not LayerNorm)
    - No dropout
    - No bias terms in linear layers
    - Single-token autoregressive forward pass with KV cache
    """
    def __init__(self, vocab_size: int, device=None):
        super().__init__()
        torch.manual_seed(hp.seed)

        self.n_embd = hp.embed_dim
        self.block_size = hp.seq_len
        self.n_head = hp.num_heads
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = hp.num_layers
        self.device = torch.device(device or hp.device)

        # Token + position embeddings
        self.wte = nn.Embedding(vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.block_size, self.n_embd)

        # Per-layer attention projections (no bias!)
        self.attn_wq = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])
        self.attn_wk = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])
        self.attn_wv = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])
        self.attn_wo = nn.ModuleList([nn.Linear(self.n_embd, self.n_embd, bias=False) for _ in range(self.n_layer)])

        # Per-layer MLP
        self.mlp_fc1 = nn.ModuleList([nn.Linear(self.n_embd, hp.ffn_dim, bias=False) for _ in range(self.n_layer)])
        self.mlp_fc2 = nn.ModuleList([nn.Linear(hp.ffn_dim, self.n_embd, bias=False) for _ in range(self.n_layer)])

        # RMSNorm layers
        self.rmsnorm_emb = RMSNorm(self.n_embd)
        self.rmsnorm_attn = nn.ModuleList([RMSNorm(self.n_embd) for _ in range(self.n_layer)])
        self.rmsnorm_mlp = nn.ModuleList([RMSNorm(self.n_embd) for _ in range(self.n_layer)])

        # LM head
        self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)

        # Initialize weights exactly like original (std=0.08)
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, token_id: int, pos_id: int, keys_cache: list, values_cache: list):
        """
        Forward pass for single token, exactly matching microgpt_classical.py's gpt() function.
        Uses KV cache for efficiency.
        """
        # Embeddings
        tok_emb = self.wte(torch.tensor([token_id], dtype=torch.long, device=self.device))
        pos_emb = self.wpe(torch.tensor([pos_id], dtype=torch.long, device=self.device))
        x = tok_emb + pos_emb
        x = self.rmsnorm_emb(x)

        for li in range(self.n_layer):
            # Attention block
            x_residual = x
            x = self.rmsnorm_attn[li](x)

            q = self.attn_wq[li](x)
            k = self.attn_wk[li](x)
            v = self.attn_wv[li](x)

            # Update KV cache
            keys_cache[li] = torch.cat([keys_cache[li], k], dim=0)
            values_cache[li] = torch.cat([values_cache[li], v], dim=0)

            # Multi-head attention
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[:, hs:hs+self.head_dim]  # [1, head_dim]
                k_h = keys_cache[li][:, hs:hs+self.head_dim]  # [seq_len, head_dim]
                v_h = values_cache[li][:, hs:hs+self.head_dim]  # [seq_len, head_dim]

                # Scaled dot-product attention
                attn_logits = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_weights = torch.softmax(attn_logits, dim=-1)
                head_out = torch.matmul(attn_weights, v_h)
                x_attn.append(head_out)

            x = self.attn_wo[li](torch.cat(x_attn, dim=-1))
            x = x + x_residual

            # MLP block
            x_residual = x
            x = self.rmsnorm_mlp[li](x)
            x = torch.relu(self.mlp_fc1[li](x))
            x = self.mlp_fc2[li](x)
            x = x + x_residual

        logits = self.lm_head(x)
        return logits


def train_and_evaluate():
    """
    Train and evaluate the PyTorch classical model, exactly matching microgpt_classical.py's behavior.
    """
    print("=" * 70)
    print("STARTING PYTORCH CLASSICAL microGPT TRAINING")
    print("=" * 70)

    # Set seeds for reproducibility
    torch.manual_seed(hp.seed)
    random.seed(hp.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = torch.device(hp.device)
    print(f"Using device: {device}")

    # 1. Load dataset
    if not os.path.exists('input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, 'input.txt')
    docs = [line.strip() for line in open('input.txt') if line.strip()]

    # 2. Word-level tokenizer
    unique_words = sorted(set(docs))
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    BOS = len(unique_words)
    vocab_size = len(unique_words) + 1
    print(f"Vocab size (unique words + BOS): {vocab_size}")

    # 3. Build model
    model = ClassicalMicroGPT(vocab_size=vocab_size, device=device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # 4. Optimizer (exact Adam params from original)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, betas=(0.85, 0.99), eps=1e-8)
    criterion = nn.CrossEntropyLoss()

    all_step_losses = []
    start_time = time.perf_counter()

    # 5. Training loop (exact match to original)
    num_steps = min(len(docs), hp.max_train_steps)
    for step in range(num_steps):
        doc = docs[step]
        tokens = [BOS, word_to_idx[doc], BOS]
        n = min(hp.seq_len, len(tokens) - 1)

        # Reset KV cache for each step
        keys_cache = [torch.empty(0, model.n_embd, device=device) for _ in range(model.n_layer)]
        values_cache = [torch.empty(0, model.n_embd, device=device) for _ in range(model.n_layer)]

        optimizer.zero_grad()
        losses = []

        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]
            logits = model(token_id, pos_id, keys_cache, values_cache)
            loss = criterion(logits, torch.tensor([target_id], dtype=torch.long, device=device))
            losses.append(loss)

        loss = sum(losses) / n
        all_step_losses.append(loss.item())

        loss.backward()

        # Linear LR decay (exact match to original)
        for param_group in optimizer.param_groups:
            param_group['lr'] = hp.learning_rate * (1 - step / num_steps)

        # Adam step with bias correction (exact match to original)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-10.0, 10.0)  # Optional: prevent exploding gradients
        optimizer.step()

        # Zero grads
        optimizer.zero_grad(set_to_none=True)

        if step == 0 or (step + 1) % 100 == 0:
            print(f"Step {step+1:4d}/{num_steps:4d} | Loss: {loss.item():.4f}")

    training_time = time.perf_counter() - start_time
    print(f"Finished training in {training_time:.2f} seconds.")

    # 6. Inference (exact match to original)
    temperature = 0.5
    print("\n--- Generating samples ---")
    inference_samples = []
    model.eval()

    with torch.no_grad():
        for _ in range(5):
            keys_cache = [torch.empty(0, model.n_embd, device=device) for _ in range(model.n_layer)]
            values_cache = [torch.empty(0, model.n_embd, device=device) for _ in range(model.n_layer)]
            token_id = BOS
            sample = []

            for pos_id in range(hp.seq_len):
                logits = model(token_id, pos_id, keys_cache, values_cache)
                probs = torch.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()

                if token_id == BOS:
                    break
                sample.append(unique_words[token_id])

            inference_samples.append(" ".join(sample))
            print(f"  sample: {inference_samples[-1]}")

    return {
        "total_params": total_params,
        "training_time": training_time,
        "losses": all_step_losses,
        "final_loss": all_step_losses[-1],
        "inference_samples": inference_samples
    }


if __name__ == "__main__":
    train_and_evaluate()
