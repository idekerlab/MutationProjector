"""Old vs. new MutationProjector on each forward() branch, with weights moved through adapt_legacy_state_dict."""
import pytest
import torch

from conftest import make_network_edges, tiny_model_kwargs
from load_model import adapt_legacy_state_dict
from MutationProjector_nn import MutationProjector as NewMutationProjector
from nn_training_functions import convert_mutations
from reference_impls import OldMutationProjector


BRANCH_CASES = [
    dict(use_special_token=True, use_representative_embedding=True, use_pooling=True),
    dict(use_special_token=True, use_representative_embedding=True, use_pooling=False),
    dict(use_special_token=False, use_representative_embedding=True, use_pooling=True),
    dict(use_special_token=False, use_representative_embedding=False, use_pooling=True),
]


@pytest.mark.parametrize("branch", BRANCH_CASES)
def test_full_forward_matches_reference(branch, tiny_model_config):
    cfg = tiny_model_config
    torch.manual_seed(42)
    network_edges = make_network_edges(cfg['num_genes'], cfg['num_networks'])
    overrides = dict(branch)
    if not branch['use_special_token']:
        overrides.update(num_special_tokens=0, num_bins=[])
    kwargs = tiny_model_kwargs(cfg, network_edges, **overrides)

    old_model = OldMutationProjector(**kwargs).eval()
    new_model = NewMutationProjector(**kwargs)
    new_model.load_state_dict(adapt_legacy_state_dict(new_model, old_model.state_dict()), strict=True)
    new_model.eval()

    torch.manual_seed(7)
    batch_size = 5
    X_converted = convert_mutations(torch.randint(0, 2, (batch_size, cfg['num_genes'], 3)))
    special = {}
    if branch['use_special_token']:
        special['X_special_tokens'] = torch.randn(batch_size, cfg['num_special_tokens'])

    with torch.no_grad():
        out1_old, _, ge_old = old_model(X_converted.clone(), test_geneset=False, return_attention_weights=False,
                                        apply_paddings=False, **special)
        out1_new, _, ge_new = new_model(X_converted.clone(), test_geneset=False, return_attention_weights=False,
                                        apply_paddings=False, **special)

    for idx, (o, n) in enumerate(zip(out1_old, out1_new)):
        assert torch.allclose(o, n, atol=1e-4), f"output1[{idx}] mismatch, branch={branch}"
    assert torch.allclose(ge_old, ge_new, atol=1e-4), f"gene_emb mismatch, branch={branch}"
