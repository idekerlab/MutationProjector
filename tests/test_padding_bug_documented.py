"""
Item 6: characterization test, not a correctness assertion. The padding path in
tokenize_special_tokens.forward instantiates a fresh nn.Embedding(1, num_features,
padding_idx=0) on every call instead of a persistent __init__ submodule -- a known,
pre-existing bug that was deliberately left unfixed (flag-only) per the approved refactor plan.

Note: because num_embeddings=1 and padding_idx=0, PyTorch always zeroes that single row right
after construction (regardless of random seed) -- so in *this* configuration the bug doesn't
manifest as randomly-varying output, it manifests as the padding embedding being permanently
stuck at all-zeros and never trainable (gradients w.r.t. it are computed but discarded every
call, since the module is never registered and never survives past the forward() call it was
created in). This test pins both symptoms as a tripwire: if a future change accidentally
"fixes" this (making padding_emb a persistent, trainable __init__ submodule), this test will
fail, so that change gets caught and the plan's flag-only decision gets revisited explicitly.
"""
import torch

from nn_training_functions import tokenize_special_tokens


def test_padding_embedding_is_always_zero_and_not_persistent():
    torch.manual_seed(0)
    tok = tokenize_special_tokens(num_features=4, num_bins=5, cuda_device=0)
    values = torch.tensor([1.0, 2.0, 3.0])

    out = tok(values, apply_padding=True)

    # symptom 1: always the zero vector (num_embeddings=1, padding_idx=0 forces this row to 0
    # on every fresh construction, regardless of seed)
    assert torch.equal(out, torch.zeros_like(out)), (
        "padding path output is no longer all-zero -- the known bug (flagged, not fixed, in "
        "this refactor) appears to have changed; if intentional, update this test and the "
        "refactor plan's item 6 decision."
    )

    # symptom 2: genuinely absent from the module's persistent, trainable parameters
    param_names = dict(tok.named_parameters()).keys()
    assert not any('padding' in name for name in param_names), (
        "a 'padding' parameter is now registered on tokenize_special_tokens -- the padding "
        "path may have been fixed to use a persistent __init__ submodule; if intentional, "
        "update this test and the refactor plan's item 6 decision."
    )
