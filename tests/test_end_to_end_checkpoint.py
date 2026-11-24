"""Real pretrained_model.pth on test_sample: old vs. new model, and new model vs. checked-in gene_emb.pt."""
from pathlib import Path

import pytest
import torch

from load_model import adapt_legacy_state_dict, load_eval_inputs, load_network_edges, production_hparams
from MutationProjector_nn import MutationProjector as NewMutationProjector
from reference_impls import OldMutationProjector

FI_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


def test_real_checkpoint_matches_reference_end_to_end():
    checkpoint = torch.load(FI_DIR / 'pretrained_model' / 'pretrained_model.pth', map_location='cpu')
    hp = production_hparams(load_network_edges(FI_DIR), device='cpu')

    old_model = OldMutationProjector(**hp)
    old_model.load_state_dict(checkpoint, strict=True)
    old_model.eval()

    new_model = NewMutationProjector(**hp)
    new_model.load_state_dict(adapt_legacy_state_dict(new_model, checkpoint), strict=True)
    new_model.eval()

    X_converted, X_special = load_eval_inputs(FI_DIR / 'data' / 'downstream_data' / 'eval_dataset' / 'test_sample')

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

    ref_gene_emb = torch.load(FI_DIR / 'prediction_results' / 'test_sample' / 'gene_emb.pt', map_location='cpu')
    assert torch.allclose(ref_gene_emb, ge_new, atol=1e-3), "gene_emb no longer matches prediction_results/test_sample"
