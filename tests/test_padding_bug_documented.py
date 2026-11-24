"""Tripwire for the unfixed padding nn.Embedding bug (refactor.md item 6).

The padding output is all zeros and has no persistent parameter. If this test fails, revisit item 6.
"""
import torch

from nn_training_functions import tokenize_special_tokens


def test_padding_embedding_is_always_zero_and_not_persistent():
    torch.manual_seed(0)
    tok = tokenize_special_tokens(num_features=4, num_bins=5, cuda_device='cpu')
    values = torch.tensor([1.0, 2.0, 3.0])

    out = tok(values, apply_padding=True)

    assert torch.equal(out, torch.zeros_like(out))
    assert not any('padding' in name for name, _ in tok.named_parameters())
