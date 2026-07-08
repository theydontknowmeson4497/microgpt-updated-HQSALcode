import pytest
import torch

from microgpt_quantum import QuantumGPT


def test_quantum_model_runs_on_cuda_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    model = QuantumGPT(vocab_size=32, use_noisy=False).to("cuda")
    token_ids = torch.tensor([1], dtype=torch.long, device="cuda")
    pos_id = 0

    keys_cache = [torch.zeros(0, model.n_embd, device="cuda") for _ in range(model.n_layer)]
    values_cache = [torch.zeros(0, model.n_embd, device="cuda") for _ in range(model.n_layer)]
    q_keys_caches = [torch.zeros(0, model.head_dim, device="cuda") for _ in range(model.n_layer * model.num_quantum_heads)]

    logits = model(token_ids, pos_id, keys_cache, values_cache, q_keys_caches)

    assert logits.shape == (32,)
