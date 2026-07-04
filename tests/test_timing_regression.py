"""
Informational, not a strict correctness gate: confirms the refactored forward pass is not
slower than the original at production scale, using the real checkpoint if available
(falls back to a large synthetic config otherwise so this still runs in CI without the
checkpoint present). This is a loose sanity bound -- the whole point of the refactor is a much
bigger speedup than "not slower", especially on GPU, but that can't be demonstrated on a
CPU-only sandbox with no batched-kernel-launch overhead to amortize.
"""
import time
from pathlib import Path

import pytest
import torch

from conftest import make_network_edges
from load_model import adapt_legacy_state_dict
from MutationProjector_nn import MutationProjector as NewMutationProjector
from nn_training_functions import convert_mutations
from reference_impls import OldMutationProjector, old_convert_mutations

pytestmark = pytest.mark.slow

FI_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = FI_DIR / 'pretrained_model' / 'pretrained_model.pth'


def _time_forward(model, X_converted, X_special, n_repeats=3):
    model.eval()
    with torch.no_grad():
        model(X_converted.clone(), X_special_tokens=X_special.clone(), test_geneset=False,
              return_attention_weights=False, apply_paddings=False)  # warmup
        t0 = time.time()
        for _ in range(n_repeats):
            model(X_converted.clone(), X_special_tokens=X_special.clone(), test_geneset=False,
                  return_attention_weights=False, apply_paddings=False)
        return (time.time() - t0) / n_repeats


def test_new_forward_not_slower_than_old_synthetic():
    torch.manual_seed(0)
    num_genes, num_features, num_networks = 60, 10, 8
    network_edges = make_network_edges(num_genes, num_networks)
    hp = dict(
        num_genes=num_genes, num_features=num_features, network_edges=network_edges,
        num_GATblock=2, num_heads=1, dropout_p=0.0, cuda_device=0, output_sizes=[3, 10, 3],
        mask_percentage=0, input_genes=[], d_ff=10, use_representative_embedding=True,
        ssl_task_index=0, use_special_token=True, num_special_tokens=9, num_bins=[5, 5, 2, 2, 2, 2, 2, 2, 2],
        use_pooling=False,
    )
    old_model = OldMutationProjector(**hp)
    new_model = NewMutationProjector(**hp)
    adapted = adapt_legacy_state_dict(new_model, old_model.state_dict())
    new_model.load_state_dict(adapted, strict=True)

    batch_size = 32
    X_raw = torch.randint(0, 2, (batch_size, num_genes, 3))
    X_converted = convert_mutations(X_raw)
    X_special = torch.randn(batch_size, 9)

    t_old = _time_forward(old_model, X_converted, X_special)
    t_new = _time_forward(new_model, X_converted, X_special)
    print(f"\nold: {t_old*1000:.1f}ms/call  new: {t_new*1000:.1f}ms/call  speedup: {t_old/t_new:.2f}x")

    # loose bound: allow generous noise margin on a shared CPU sandbox, this is informational
    assert t_new < t_old * 1.5, f"new forward pass ({t_new:.3f}s) is unexpectedly slower than old ({t_old:.3f}s)"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="real pretrained_model.pth not present")
def test_new_forward_not_slower_than_old_real_checkpoint():
    from test_end_to_end_checkpoint import _production_hparams, NETWORKS, NETWORK_DIR, TEST_SAMPLE_DIR
    import pandas as pd
    from nn_training_functions import merge_data

    network_edges = [torch.load(NETWORK_DIR / f'{n}.pt', map_location='cpu') for n in NETWORKS]
    tmp = torch.load(CHECKPOINT, map_location='cpu')
    hp = _production_hparams(network_edges)

    old_model = OldMutationProjector(**hp)
    old_model.load_state_dict(tmp, strict=True)
    new_model = NewMutationProjector(**hp)
    new_model.load_state_dict(adapt_legacy_state_dict(new_model, tmp), strict=True)

    mdf = pd.read_csv(TEST_SAMPLE_DIR / 'mut.txt', sep='\t')
    cna = pd.read_csv(TEST_SAMPLE_DIR / 'cna.txt', sep='\t')
    cnd = pd.read_csv(TEST_SAMPLE_DIR / 'cnd.txt', sep='\t')
    X_test, _, _ = merge_data(mdf, cna, cnd, use_cancer_types=False)
    covariates = pd.read_csv(TEST_SAMPLE_DIR / 'covariates.txt', sep='\t')
    X_special = torch.tensor(covariates.set_index('sample').values)
    X_converted = convert_mutations(X_test)

    t_old = _time_forward(old_model, X_converted, X_special)
    t_new = _time_forward(new_model, X_converted, X_special)
    print(f"\n[real checkpoint] old: {t_old*1000:.1f}ms/call  new: {t_new*1000:.1f}ms/call  speedup: {t_old/t_new:.2f}x")

    assert t_new < t_old * 1.5, f"new forward pass ({t_new:.3f}s) is unexpectedly slower than old ({t_old:.3f}s)"
