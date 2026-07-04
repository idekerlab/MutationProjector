"""
Item 5: new (torch.bucketize) vs. old (pandas.cut) binning inside tokenize_special_tokens.
pd.cut(bins=int) computes adaptive, batch-dependent bin edges (min/max of the values passed
in *this* call, widened by 0.1% of the range, right-closed intervals) -- this is the one
refactor step where equivalence isn't structurally obvious, so it gets the widest randomized
corpus of any test in this suite, including deliberately-adversarial exact-boundary values.
"""
import numpy as np
import pytest
import torch

from nn_training_functions import tokenize_special_tokens
from reference_impls import OldTokenizeSpecialTokens


def _make_pair(num_features, num_bins, seed):
    torch.manual_seed(seed)
    new = tokenize_special_tokens(num_features, num_bins, cuda_device=0)
    torch.manual_seed(seed)
    old = OldTokenizeSpecialTokens(num_features, num_bins, cuda_device=0)
    assert torch.equal(new.token_emb.weight, old.token_emb.weight)
    return new, old


def _values(kind, n, rng):
    if kind == 'normal':
        return rng.randn(n) * rng.uniform(0.1, 100)
    if kind == 'all_equal':
        return np.array([rng.choice([0.0, 5.0, -3.0])] * n)
    if kind == 'ints':
        return rng.randint(0, 5, size=n).astype(float)
    if kind == 'wide_range':
        return rng.uniform(-1e6, 1e6, n)
    if kind == 'single':
        return rng.uniform(-10, 10, 1)
    raise ValueError(kind)


@pytest.mark.parametrize("trial", range(60))
def test_matches_pandas_cut_across_input_kinds_and_types(trial):
    rng = np.random.RandomState(trial)
    num_features = rng.randint(2, 8)
    num_bins = rng.randint(2, 9)
    n = rng.randint(1, 40)
    kind = rng.choice(['normal', 'all_equal', 'ints', 'wide_range', 'single'])
    arr = _values(kind, n, rng)
    input_type = rng.choice(['list', 'ndarray', 'tensor'])
    values = arr.tolist() if input_type == 'list' else (arr if input_type == 'ndarray' else torch.tensor(arr))

    new, old = _make_pair(num_features, num_bins, seed=trial)
    new.eval(); old.eval()

    with torch.no_grad():
        out_new = new(values, apply_padding=False)
        out_old = old(values, apply_padding=False)

    assert torch.allclose(out_old, out_new, atol=1e-6), f"trial={trial} kind={kind} input_type={input_type}"


@pytest.mark.parametrize("trial", range(200))
def test_matches_pandas_cut_at_exact_bin_boundaries(trial):
    """Adversarial: values placed exactly on a computed bin edge, where naive torch.linspace
    (vs. numpy's) can disagree with pandas by float64 ULPs -- this is why the implementation
    computes edges via np.linspace rather than torch.linspace."""
    rng = np.random.RandomState(trial)
    num_bins = rng.randint(2, 12)
    mn, mx = sorted(rng.uniform(-50, 50, 2))
    if mx - mn < 1e-6:
        mx = mn + 1
    edges = np.linspace(mn, mx, num_bins + 1)
    vals = np.concatenate([edges, rng.uniform(mn, mx, 10)])

    new, old = _make_pair(num_features=3, num_bins=num_bins, seed=trial)
    with torch.no_grad():
        out_new = new(vals, apply_padding=False)
        out_old = old(vals, apply_padding=False)
    assert torch.allclose(out_old, out_new, atol=1e-6), f"trial={trial}"
