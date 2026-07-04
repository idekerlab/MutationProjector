"""
Items 3+4: old (PyG Data/DataLoader-based, pandas GATidx lookup) vs. new (2D-indexed
GAT_layers, cached block-diagonal offset edge_index) GATv2block, with identical weights.
"""
import torch

from GATv2_functions import GATv2block as NewGATv2block
from reference_impls import OldGATv2block
from compute_attention import compute_attention
from conftest import make_network_edges


def copy_weights(old, new, num_GATblock, num_networks):
    for i in range(num_GATblock):
        for j in range(num_networks):
            idx = i * num_networks + j
            new.GAT_layers[i][j].load_state_dict(old.GAT_layers[idx].state_dict())
    new.linear_layers.weight.data.copy_(torch.stack([old.linear_layers[g].weight for g in range(old.num_genes)]))
    new.linear_layers.bias.data.copy_(torch.stack([old.linear_layers[g].bias for g in range(old.num_genes)]))
    new.FF_layer1.weight.data.copy_(torch.stack([old.FF_layer1[g].weight for g in range(old.num_genes)]))
    new.FF_layer1.bias.data.copy_(torch.stack([old.FF_layer1[g].bias for g in range(old.num_genes)]))
    new.FF_layer2.weight.data.copy_(torch.stack([old.FF_layer2[g].weight for g in range(old.num_genes)]))
    new.FF_layer2.bias.data.copy_(torch.stack([old.FF_layer2[g].bias for g in range(old.num_genes)]))
    new.layer_norm1.load_state_dict(old.layer_norm1.state_dict())
    new.layer_norm2.load_state_dict(old.layer_norm2.state_dict())


CASES = [
    dict(num_genes=6, num_features=4, num_networks=2, num_GATblock=2, batch_size=3, num_heads=1, seed=0),
    dict(num_genes=10, num_features=3, num_networks=3, num_GATblock=1, batch_size=5, num_heads=1, seed=1),
    dict(num_genes=4, num_features=5, num_networks=1, num_GATblock=3, batch_size=1, num_heads=1, seed=2),
    dict(num_genes=8, num_features=4, num_networks=4, num_GATblock=2, batch_size=7, num_heads=1, seed=3),
    dict(num_genes=23, num_features=10, num_networks=8, num_GATblock=2, batch_size=4, num_heads=1, seed=4),  # production-shaped
]


def _run_case(num_genes, num_features, num_networks, num_GATblock, batch_size, num_heads, seed):
    torch.manual_seed(seed)
    network_edges = make_network_edges(num_genes, num_networks)

    old = OldGATv2block(num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p=0.0, cuda_device='cpu', d_ff=7, self_loop=False)
    new = NewGATv2block(num_genes, num_features, network_edges, num_GATblock, num_heads, dropout_p=0.0, cuda_device='cpu', d_ff=7, self_loop=False)
    copy_weights(old, new, num_GATblock, num_networks)
    old.eval(); new.eval()

    X_input = torch.randn(batch_size, num_genes, num_features)
    with torch.no_grad():
        X_old, edges_old, attn_old = old(X_input.clone(), True)
        X_new, edges_new, attn_new = new(X_input.clone(), True)
    return X_old, X_new, edges_old, edges_new, attn_old, attn_new


def test_output_matches_reference():
    for case in CASES:
        X_old, X_new, *_ = _run_case(**case)
        assert torch.allclose(X_old, X_new, atol=1e-5), case


def test_edge_index_and_attention_match_reference():
    for case in CASES:
        _, _, edges_old, edges_new, attn_old, attn_new = _run_case(**case)
        for key in edges_old:
            assert torch.equal(edges_old[key], edges_new[key]), (case, key)
            assert torch.allclose(attn_old[key], attn_new[key], atol=1e-5), (case, key)


def test_compute_attention_consumer_matches_reference():
    # compute_attention.py assumes edge_index encodes sample_idx via node_idx // num_genes;
    # this is the actual downstream consumer of the batching-order convention item 4 relies on.
    import numpy as np
    for case in CASES:
        num_genes, batch_size = case['num_genes'], case['batch_size']
        _, _, edges_old, edges_new, attn_old, attn_new = _run_case(**case)
        X1_dummy = torch.zeros(batch_size, num_genes)
        X2_dummy = torch.zeros(batch_size, 0)
        ca_old = compute_attention(X1_dummy, X2_dummy, edges_old, attn_old)
        ca_new = compute_attention(X1_dummy, X2_dummy, edges_new, attn_new)
        assert np.allclose(ca_old, ca_new, equal_nan=True), case
