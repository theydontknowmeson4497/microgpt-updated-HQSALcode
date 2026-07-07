import torch
import torch.nn as nn
import math
from model import FeedForwardNetwork

class ClassicalSelfAttention(nn.Module):
    """
    Standard Classical Multi-Head Self-Attention Layer.
    Used as a baseline for ablation studies.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        batch_size, seq_len, _ = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Split into heads: [B, num_heads, L, d_head]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # 3. Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores + mask
            
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # 4. Weighted values
        out = torch.matmul(weights, v) # [B, num_heads, L, d_head]
        
        # 5. Concatenate and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(out)
        
        return output, weights

class ClassicalTransformerDecoderLayer(nn.Module):
    """
    Standard classical transformer decoder layer.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ClassicalSelfAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        norm_x = self.norm1(x)
        attn_out, weights = self.attn(norm_x, mask=mask)
        x = x + self.dropout1(attn_out)
        
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout2(ffn_out)
        
        return x, weights

class ClassicalTransformerDecoder(nn.Module):
    """
    Standard classical transformer decoder language model.
    """
    def __init__(self, vocab_size: int, embed_dim: int, seq_len: int, num_heads: int,
                 ffn_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            ClassicalTransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, tokens: torch.Tensor) -> tuple:
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.seq_len
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=tokens.device).unsqueeze(0)
        x = self.token_embeddings(tokens) + self.position_embeddings(pos)
        x = self.dropout(x)
        
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=tokens.device), diagonal=1)
        
        all_attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask=mask)
            all_attn_weights.append(weights)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits, all_attn_weights
