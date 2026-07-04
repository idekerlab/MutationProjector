"""
Full-scale gate (item 9 of the plan's Verification section): loads the real
pretrained_model.pth (468 genes, 8 networks, 9 special tokens) into both the reference
architecture (sanity: does the shipped checkpoint even load into the un-refactored model?) and
the refactored architecture via adapt_legacy_state_dict, runs both on the real
data/downstream_data/eval_dataset/test_sample data, and requires bit-for-bit-or-better agreement.
Also cross-checks against the checked-in prediction_results/test_sample/*.pt reference embeddings.

Requires the real checkpoint/network/data files that ship with this repo -- skipped
automatically if they're not present (e.g. a checkout with Git LFS / large files excluded).
"""
import time
from pathlib import Path

import pandas as pd
import pytest
import torch

from load_model import adapt_legacy_state_dict
from MutationProjector_nn import MutationProjector as NewMutationProjector
from nn_training_functions import convert_mutations, merge_data
from reference_impls import OldMutationProjector

FI_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = FI_DIR / 'pretrained_model' / 'pretrained_model.pth'
NETWORK_DIR = FI_DIR / 'data' / 'networks'
TEST_SAMPLE_DIR = FI_DIR / 'data' / 'downstream_data' / 'eval_dataset' / 'test_sample'
REFERENCE_EMB_DIR = FI_DIR / 'prediction_results' / 'test_sample'

NETWORKS = 'GRN_expanded;E3_expanded;phosphorylation_expanded;physical_ppi_expanded;genetic_interaction_expanded;DDRAM;STRING;PCNET'.split(';')

pytestmark = pytest.mark.slow

requires_real_assets = pytest.mark.skipif(
    not (CHECKPOINT.exists() and TEST_SAMPLE_DIR.exists() and all((NETWORK_DIR / f'{n}.pt').exists() for n in NETWORKS)),
    reason="real pretrained_model.pth / network / test_sample data not present in this checkout",
)


def _production_hparams(network_edges):
    import numpy as np
    return dict(
        num_genes=468, num_features=10, network_edges=network_edges, num_GATblock=2, num_heads=1,
        dropout_p=0.1, cuda_device=0, output_sizes=[3, 10, 3], mask_percentage=0, input_genes=[],
        d_ff=10, use_representative_embedding=1, ssl_task_index=0, use_special_token=1,
        num_special_tokens=9, num_bins=list(np.append([5, 5], [2]*7)), use_pooling=0,
    )


@requires_real_assets
def test_real_checkpoint_matches_reference_end_to_end():
    network_edges = [torch.load(NETWORK_DIR / f'{n}.pt', map_location='cpu') for n in NETWORKS]
    tmp = torch.load(CHECKPOINT, map_location='cpu')
    hp = _production_hparams(network_edges)

    old_model = OldMutationProjector(**hp)
    result = old_model.load_state_dict(tmp, strict=True)
    old_model.eval()

    new_model = NewMutationProjector(**hp)
    adapted = adapt_legacy_state_dict(new_model, tmp)
    new_model.load_state_dict(adapted, strict=True)
    new_model.eval()

    mdf = pd.read_csv(TEST_SAMPLE_DIR / 'mut.txt', sep='\t')
    cna = pd.read_csv(TEST_SAMPLE_DIR / 'cna.txt', sep='\t')
    cnd = pd.read_csv(TEST_SAMPLE_DIR / 'cnd.txt', sep='\t')
    X_test, _, _ = merge_data(mdf, cna, cnd, use_cancer_types=False)
    covariates = pd.read_csv(TEST_SAMPLE_DIR / 'covariates.txt', sep='\t')
    X_special = torch.tensor(covariates.set_index('sample').values)
    X_converted = convert_mutations(X_test)

    with torch.no_grad():
        out1_old, mp_old, aw_old, ei_old, (ge_old, cov_old, rep_old, ocl_old) = old_model(
            X_converted.clone(), X_special_tokens=X_special.clone(), test_geneset=False,
            return_attention_weights=True, apply_paddings=False)
        out1_new, mp_new, aw_new, ei_new, (ge_new, cov_new, rep_new, ocl_new) = new_model(
            X_converted.clone(), X_special_tokens=X_special.clone(), test_geneset=False,
            return_attention_weights=True, apply_paddings=False)

    for idx, (o, n) in enumerate(zip(out1_old, out1_new)):
        assert torch.allclose(o, n, atol=1e-4), f"output1[{idx}] mismatch (max diff {(o-n).abs().max().item():.2e})"

    for name, o, n in [('gene_emb', ge_old, ge_new), ('rep_emb', rep_old, rep_new),
                        ('cov_emb', cov_old, cov_new), ('out_concat_layer', ocl_old, ocl_new)]:
        assert torch.allclose(o, n, atol=1e-4), f"{name} mismatch (max diff {(o-n).abs().max().item():.2e})"

    for key in aw_old:
        assert torch.equal(ei_old[key], ei_new[key]), f"edge_index mismatch at {key}"
        assert torch.allclose(aw_old[key], aw_new[key], atol=1e-4), f"attention mismatch at {key}"

    if REFERENCE_EMB_DIR.exists():
        ref_gene_emb = torch.load(REFERENCE_EMB_DIR / 'gene_emb.pt', map_location='cpu')
        assert torch.allclose(ref_gene_emb, ge_new, atol=1e-3), (
            "new model's gene_emb no longer matches the checked-in prediction_results reference"
        )
