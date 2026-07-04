"""
Full-model integration: old (pre-refactor) vs. new (refactored, all 8 items combined)
MutationProjector, across the branches of forward() that item 7's redundant-.cuda() removals
and item 8's SSL-head vectorization touch (use_special_token, use_representative_embedding,
use_pooling on/off). Weights are transferred old->new via adapt_legacy_state_dict so this is
also a checkpoint-bridge smoke test at every branch combination, not just the default one.
"""
import pytest
import torch

from conftest import make_network_edges
from load_model import adapt_legacy_state_dict
from MutationProjector_nn import MutationProjector as NewMutationProjector
from nn_training_functions import convert_mutations
from reference_impls import OldMutationProjector, old_convert_mutations


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

    common = dict(
        num_genes=cfg['num_genes'], num_features=cfg['num_features'], network_edges=network_edges,
        num_GATblock=cfg['num_GATblock'], num_heads=cfg['num_heads'], dropout_p=0.0, cuda_device='cpu',
        output_sizes=cfg['output_sizes'], mask_percentage=0, input_genes=[], d_ff=cfg['d_ff'],
        ssl_task_index=cfg['ssl_task_index'],
    )
    if branch['use_special_token']:
        common.update(use_special_token=True, num_special_tokens=cfg['num_special_tokens'], num_bins=cfg['num_bins'])
    else:
        common.update(use_special_token=False, num_special_tokens=0, num_bins=[])
    common.update(use_representative_embedding=branch['use_representative_embedding'], use_pooling=branch['use_pooling'])

    old_model = OldMutationProjector(**common)
    old_model.eval()
    old_state = old_model.state_dict()

    new_model = NewMutationProjector(**common)
    adapted = adapt_legacy_state_dict(new_model, old_state)
    result = new_model.load_state_dict(adapted, strict=True)
    assert list(result.missing_keys) == [] and list(result.unexpected_keys) == []
    new_model.eval()

    torch.manual_seed(7)
    batch_size = 5
    X_raw = torch.randint(0, 2, (batch_size, cfg['num_genes'], 3))
    X_converted_old = old_convert_mutations(X_raw)
    X_converted_new = convert_mutations(X_raw)
    assert torch.equal(X_converted_old, X_converted_new)

    kwargs = {}
    if branch['use_special_token']:
        X_special = torch.randn(batch_size, cfg['num_special_tokens'])
        kwargs['X_special_tokens'] = X_special

    # Pre-existing bug in the original code (unrelated to this refactor, reproduced identically
    # by both old and new): with use_special_token=False and use_representative_embedding=False,
    # `out_concat_layer` is never assigned, so return_attention_weights=True raises
    # UnboundLocalError in both. Use the plain 3-tuple return for that one branch instead.
    want_attention = not (branch['use_special_token'] is False and branch['use_representative_embedding'] is False)

    with torch.no_grad():
        if want_attention:
            out1_old, mp_old, aw_old, ei_old, extras_old = old_model(
                X_converted_old.clone(), test_geneset=False, return_attention_weights=True,
                apply_paddings=False, **kwargs)
            out1_new, mp_new, aw_new, ei_new, extras_new = new_model(
                X_converted_new.clone(), test_geneset=False, return_attention_weights=True,
                apply_paddings=False, **kwargs)
        else:
            out1_old, mp_old, ge_old = old_model(
                X_converted_old.clone(), test_geneset=False, return_attention_weights=False,
                apply_paddings=False, **kwargs)
            out1_new, mp_new, ge_new = new_model(
                X_converted_new.clone(), test_geneset=False, return_attention_weights=False,
                apply_paddings=False, **kwargs)
            aw_old, extras_old = {}, (ge_old,)
            aw_new, extras_new = {}, (ge_new,)

    for idx, (o, n) in enumerate(zip(out1_old, out1_new)):
        assert torch.allclose(o, n, atol=1e-4), f"output1[{idx}] mismatch, branch={branch}"

    names = ['gene_emb', 'cov_emb', 'rep_emb', 'out_concat_layer'] if want_attention else ['gene_emb']
    for name, o, n in zip(names, extras_old, extras_new):
        if o.numel() == 0:
            continue
        assert torch.allclose(o, n, atol=1e-4), f"{name} mismatch, branch={branch}"

    for key in aw_old:
        assert torch.equal(ei_old[key], ei_new[key]), (branch, key)
        assert torch.allclose(aw_old[key], aw_new[key], atol=1e-4), (branch, key)
