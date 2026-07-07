import torch
import torch.nn as nn
from quantum_attention import HybridQuantumAttention

class FeedForwardNetwork(nn.Module):
    """
    Standard Feed-Forward Network (FFN) block used in transformer layers.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridQuantumTransformerDecoderLayer(nn.Module):
    """
    A single Transformer decoder layer combining a Hybrid Quantum Self-Attention
    block and a classical Feed-Forward Network. Uses Pre-Layer Normalization.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_quantum_heads: int,
                 num_qubits: int, q_depth: int, ffn_dim: int, 
                 use_noisy_simulator: bool = False, shots: int = 1024, 
                 depol_error: float = 0.01, use_gpu_simulator: bool = False,
                 dropout: float = 0.1, entangler: str = "ring"):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = HybridQuantumAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_quantum_heads=num_quantum_heads,
            num_qubits=num_qubits,
            q_depth=q_depth,
            use_noisy_simulator=use_noisy_simulator,
            shots=shots,
            depol_error=depol_error,
            use_gpu_simulator=use_gpu_simulator,
            dropout=dropout,
            entangler=entangler
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        # Pre-LN attention block
        norm_x = self.norm1(x)
        attn_out, weights = self.attn(norm_x, mask=mask)
        x = x + self.dropout1(attn_out)
        
        # Pre-LN FFN block
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout2(ffn_out)
        
        return x, weights

class HybridQuantumTransformerDecoder(nn.Module):
    """
    Complete language model based on Hybrid Quantum Self-Attention.
    """
    def __init__(self, vocab_size: int, embed_dim: int, seq_len: int, num_heads: int,
                 num_quantum_heads: int, num_qubits: int, q_depth: int, ffn_dim: int,
                 num_layers: int = 1, use_noisy_simulator: bool = False,
                 shots: int = 1024, depol_error: float = 0.01, use_gpu_simulator: bool = False,
                 dropout: float = 0.1, entangler: str = "ring"):
        super().__init__()
        
        self.seq_len = seq_len
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Stack decoder layers
        self.layers = nn.ModuleList([
            HybridQuantumTransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_quantum_heads=num_quantum_heads,
                num_qubits=num_qubits,
                q_depth=q_depth,
                ffn_dim=ffn_dim,
                use_noisy_simulator=use_noisy_simulator,
                shots=shots,
                depol_error=depol_error,
                use_gpu_simulator=use_gpu_simulator,
                dropout=dropout,
                entangler=entangler
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tokens: torch.Tensor) -> tuple:
        # tokens shape: [batch_size, seq_len]
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.seq_len, f"Sequence length {seq_len} exceeds max length {self.seq_len}"
        
        # Compute embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0)
        x = self.token_embeddings(tokens) + self.position_embeddings(pos)
        x = self.dropout(x)
        
        # Create causal upper-triangular mask
        # 0 where we can attend, -inf where we cannot
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=tokens.device), diagonal=1)
        
        # Forward through layers
        all_attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask=mask)
            all_attn_weights.append(weights)
            
        x = self.final_norm(x)
        logits = self.lm_head(x) # [batch_size, seq_len, vocab_size]
        
        return logits, all_attn_weights
