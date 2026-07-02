import torch
import torch.nn as nn
from config import Hyperparameters as hp
from quantum_layer import QuantumFeatureExtractor
from quantum_attention import HybridQuantumAttention
from model import HybridQuantumTransformerDecoder
from utils import set_seed, count_parameters

def test_quantum_feature_extractor():
    print("Testing QuantumFeatureExtractor...")
    set_seed(42)
    batch_size = 2
    seq_len = 8
    num_qubits = hp.num_qubits
    
    extractor = QuantumFeatureExtractor(
        num_qubits=num_qubits, 
        q_depth=2,
        use_noisy_simulator=False
    )
    
    # Random input matching [batch_size, seq_len, num_qubits]
    x = torch.randn(batch_size, seq_len, num_qubits)
    out = extractor(x)
    
    # Assert output shape matches input shape
    assert out.shape == (batch_size, seq_len, num_qubits), f"Expected shape {(batch_size, seq_len, num_qubits)}, got {out.shape}"
    print("  [PASSED] Forward output shape matches input shape.")
    
    # Assert gradients flow back to the quantum weights
    loss = out.sum()
    loss.backward()
    assert extractor.qnn_layer.weight.grad is not None, "Gradients did not flow to the quantum weights!"
    assert not torch.allclose(extractor.qnn_layer.weight.grad, torch.zeros_like(extractor.qnn_layer.weight.grad)), "Quantum gradients are all zeros!"
    print("  [PASSED] Backpropagation and gradient calculations succeed.")

def test_hybrid_quantum_attention():
    print("Testing HybridQuantumAttention...")
    set_seed(42)
    batch_size = 2
    seq_len = 8
    
    attn = HybridQuantumAttention(
        embed_dim=hp.embed_dim,
        num_heads=hp.num_heads,
        num_quantum_heads=hp.num_quantum_heads,
        num_qubits=hp.num_qubits,
        q_depth=hp.q_depth,
        use_noisy_simulator=False,
        dropout=hp.dropout
    )
    
    x = torch.randn(batch_size, seq_len, hp.embed_dim)
    
    # Test without mask
    out, weights = attn(x)
    assert out.shape == (batch_size, seq_len, hp.embed_dim), f"Expected out shape {(batch_size, seq_len, hp.embed_dim)}, got {out.shape}"
    assert weights.shape == (batch_size, hp.num_heads, seq_len, seq_len), f"Expected weights shape {(batch_size, hp.num_heads, seq_len, seq_len)}, got {weights.shape}"
    print("  [PASSED] Forward attention and weight tensor shapes match (without mask).")
    
    # Test with causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    out_masked, weights_masked = attn(x, mask=mask)
    
    # Verify upper-triangular masking worked (weights should be 0 on masked indices)
    # The upper triangular portion (excluding diagonal) of each attention map must be zero
    for b in range(batch_size):
        for h in range(hp.num_heads):
            # Extract upper triangle (above diagonal) of attention weights
            upper_tri = torch.triu(weights_masked[b, h], diagonal=1)
            assert torch.allclose(upper_tri, torch.zeros_like(upper_tri)), f"Masking failed for head {h} batch {b}!"
            
    print("  [PASSED] Forward attention and masking correctness.")

def test_full_hybrid_model():
    print("Testing HybridQuantumTransformerDecoder...")
    set_seed(42)
    batch_size = 2
    seq_len = 16
    
    model = HybridQuantumTransformerDecoder(
        vocab_size=hp.vocab_size,
        embed_dim=hp.embed_dim,
        seq_len=seq_len, # Max sequence length
        num_heads=hp.num_heads,
        num_quantum_heads=hp.num_quantum_heads,
        num_qubits=hp.num_qubits,
        q_depth=hp.q_depth,
        ffn_dim=hp.ffn_dim,
        num_layers=hp.num_layers,
        use_noisy_simulator=False,
        dropout=hp.dropout
    )
    
    # Mock inputs: list of integer tokens of size [batch_size, seq_len]
    tokens = torch.randint(1, hp.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, attn_weights = model(tokens)
    
    # Assert logits shape is [batch_size, seq_len, vocab_size]
    assert logits.shape == (batch_size, seq_len, hp.vocab_size), f"Expected shape {(batch_size, seq_len, hp.vocab_size)}, got {logits.shape}"
    assert len(attn_weights) == hp.num_layers, f"Expected {hp.num_layers} layer weight matrices, got {len(attn_weights)}"
    print("  [PASSED] Full model forward pass and output logits shape verify.")
    
    # Loss computation & backward pass sanity check
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(1, hp.vocab_size, (batch_size, seq_len))
    loss = criterion(logits.view(-1, hp.vocab_size), target.view(-1))
    loss.backward()
    
    # Check that model weights have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} does not have gradients!"
            
    print("  [PASSED] Loss backward pass and global parameter gradient checks.")

def test_noisy_simulation_setup():
    print("Testing Noisy/Shot-Based Simulation Setup...")
    set_seed(42)
    
    # Short test to ensure noisy simulation works without throwing exceptions
    extractor = QuantumFeatureExtractor(
        num_qubits=hp.num_qubits, 
        q_depth=1,
        use_noisy_simulator=True,
        shots=10, # Very small number of shots for fast test
        depol_error=0.05
    )
    
    x = torch.randn(1, 4, hp.num_qubits)
    out = extractor(x)
    assert out.shape == (1, 4, hp.num_qubits)
    
    loss = out.sum()
    loss.backward()
    assert extractor.qnn_layer.weight.grad is not None
    print("  [PASSED] Noisy shot-based simulator forward and backward pass.")

def run_all_tests():
    print("=" * 60)
    print("RUNNING HYBRID QUANTUM SELF-ATTENTION SANITY CHECKS")
    print("=" * 60)
    test_quantum_feature_extractor()
    print("-" * 50)
    test_hybrid_quantum_attention()
    print("-" * 50)
    test_full_hybrid_model()
    print("-" * 50)
    test_noisy_simulation_setup()
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
