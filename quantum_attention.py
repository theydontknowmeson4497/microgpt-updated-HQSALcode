import torch
import torch.nn as nn
import math
from quantum_layer import QuantumFeatureExtractor

class HybridQuantumAttention(nn.Module):
    """
    Hybrid Quantum Self-Attention Layer.
    Splits the embedding into multiple heads. For the designated quantum heads,
    the Query (Q) and Key (K) vectors are projected down, passed through a 
    Parameterized Quantum Circuit (VQC), projected back, and then compared 
    classically. The other heads are processed classically.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_quantum_heads: int,
                 num_qubits: int, q_depth: int, use_noisy_simulator: bool = False,
                 shots: int = 1024, depol_error: float = 0.01, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.num_quantum_heads = num_quantum_heads
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Quantum heads configuration
        # For each quantum head, we define classical projection adapters to map d_head -> num_qubits
        # and back num_qubits -> d_head.
        self.q_extractors = nn.ModuleList()
        self.k_extractors = nn.ModuleList()
        self.q_in_projs = nn.ModuleList()
        self.q_out_projs = nn.ModuleList()
        self.k_in_projs = nn.ModuleList()
        self.k_out_projs = nn.ModuleList()
        
        for h in range(num_quantum_heads):
            # QNN layers for Query and Key
            self.q_extractors.append(
                QuantumFeatureExtractor(
                    num_qubits=num_qubits,
                    q_depth=q_depth,
                    use_noisy_simulator=use_noisy_simulator,
                    shots=shots,
                    depol_error=depol_error
                )
            )
            self.k_extractors.append(
                QuantumFeatureExtractor(
                    num_qubits=num_qubits,
                    q_depth=q_depth,
                    use_noisy_simulator=use_noisy_simulator,
                    shots=shots,
                    depol_error=depol_error
                )
            )
            # Classical adapters
            self.q_in_projs.append(nn.Linear(self.d_head, num_qubits, bias=False))
            self.q_out_projs.append(nn.Linear(num_qubits, self.d_head, bias=False))
            self.k_in_projs.append(nn.Linear(self.d_head, num_qubits, bias=False))
            self.k_out_projs.append(nn.Linear(num_qubits, self.d_head, bias=False))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        # x shape: [batch_size, seq_len, embed_dim]
        # mask shape: [seq_len, seq_len] or [batch_size, 1, seq_len, seq_len]
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(x) # [B, L, D]
        k = self.k_proj(x) # [B, L, D]
        v = self.v_proj(x) # [B, L, D]
        
        # 2. Reshape and transpose to split heads: [B, num_heads, L, d_head]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # We will accumulate output states for each head
        head_outputs = []
        # Store attention weights for inspection/visualization
        attn_weights_list = []
        
        # 3. Process each head
        for h in range(self.num_heads):
            q_h = q[:, h, :, :]  # [B, L, d_head]
            k_h = k[:, h, :, :]  # [B, L, d_head]
            v_h = v[:, h, :, :]  # [B, L, d_head]
            
            if h < self.num_quantum_heads:
                # --- Quantum Head Processing ---
                # A. Project d_head down to num_qubits classically
                q_qiskit_in = self.q_in_projs[h](q_h)  # [B, L, num_qubits]
                k_qiskit_in = self.k_in_projs[h](k_h)  # [B, L, num_qubits]
                
                # B. Pass through Variational Quantum Circuits (QNN)
                q_qiskit_out = self.q_extractors[h](q_qiskit_in)  # [B, L, num_qubits]
                k_qiskit_out = self.k_extractors[h](k_qiskit_in)  # [B, L, num_qubits]
                
                # C. Project back to d_head classically
                q_h_prime = self.q_out_projs[h](q_qiskit_out)  # [B, L, d_head]
                k_h_prime = self.k_out_projs[h](k_qiskit_out)  # [B, L, d_head]
            else:
                # --- Classical Head Processing ---
                q_h_prime = q_h
                k_h_prime = k_h
                
            # D. Compute scaled dot-product attention scores
            # scores shape: [B, L, L]
            scores = torch.matmul(q_h_prime, k_h_prime.transpose(-2, -1)) / math.sqrt(self.d_head)
            
            # E. Apply causal mask if provided
            if mask is not None:
                # Add mask (mask has 0 for allowed, -inf for masked out)
                scores = scores + mask
                
            # F. Softmax to obtain attention weights
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            attn_weights_list.append(weights.unsqueeze(1)) # Keep track of head weight [B, 1, L, L]
            
            # G. Weighted sum of values
            # out_h shape: [B, L, d_head]
            out_h = torch.matmul(weights, v_h)
            head_outputs.append(out_h)
            
        # 4. Concatenate heads back: [B, L, D]
        # head_outputs list of length num_heads, each [B, L, d_head]
        concat_out = torch.cat(head_outputs, dim=-1)
        
        # 5. Final output projection
        output = self.out_proj(concat_out)
        
        # Combine attention weights across heads: [B, num_heads, L, L]
        all_attn_weights = torch.cat(attn_weights_list, dim=1)
        
        return output, all_attn_weights
