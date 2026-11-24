"""torch.bucketize binning in tokenize_special_tokens vs. pd.cut, which sets bin edges from each batch's min and max."""
import numpy as np
import torch

from nn_training_functions import tokenize_special_tokens
from reference_impls import OldTokenizeSpecialTokens
from conftest import seeded_pair


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


def test_matches_pandas_cut_across_input_kinds_and_types():
    failures = []
    for trial in range(60):
        rng = np.random.RandomState(trial)
        num_features = rng.randint(2, 8)
        num_bins = rng.randint(2, 9)
        n = rng.randint(1, 40)
        kind = rng.choice(['normal', 'all_equal', 'ints', 'wide_range', 'single'])
        arr = _values(kind, n, rng)
        input_type = rng.choice(['list', 'ndarray', 'tensor'])
        values = arr.tolist() if input_type == 'list' else (arr if input_type == 'ndarray' else torch.tensor(arr))

        new, old = seeded_pair(trial, tokenize_special_tokens, OldTokenizeSpecialTokens, num_features, num_bins, cuda_device='cpu')
        new.eval(); old.eval()
        with torch.no_grad():
            out_new = new(values, apply_padding=False)
            out_old = old(values, apply_padding=False)
        if not torch.allclose(out_old, out_new, atol=1e-6):
            failures.append(dict(trial=trial, kind=kind, input_type=input_type))
    assert failures == []


def test_matches_pandas_cut_at_exact_bin_boundaries():
    """Values on a computed bin edge, where torch.linspace and np.linspace can differ by float64 ULPs."""
    failures = []
    for trial in range(200):
        rng = np.random.RandomState(trial)
        num_bins = rng.randint(2, 12)
        mn, mx = sorted(rng.uniform(-50, 50, 2))
        if mx - mn < 1e-6:
            mx = mn + 1
        edges = np.linspace(mn, mx, num_bins + 1)
        vals = np.concatenate([edges, rng.uniform(mn, mx, 10)])

        new, old = seeded_pair(trial, tokenize_special_tokens, OldTokenizeSpecialTokens, 3, num_bins, cuda_device='cpu')
        with torch.no_grad():
            out_new = new(vals, apply_padding=False)
            out_old = old(vals, apply_padding=False)
        if not torch.allclose(out_old, out_new, atol=1e-6):
            failures.append(dict(trial=trial, num_bins=num_bins))
    assert failures == []
