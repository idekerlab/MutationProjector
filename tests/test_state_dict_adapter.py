"""adapt_legacy_state_dict: legacy per-gene and flat GAT_layers keys map to the batched, 2D-indexed model."""
import torch

from conftest import make_network_edges, tiny_model_kwargs
from load_model import adapt_legacy_state_dict
from MutationProjector_nn import MutationProjector


def build_legacy_state_dict(num_genes, num_features, num_networks, num_GATblock, d_ff, output_sizes, ssl_task_index):
    sd = {}
    # GATblock per-gene linears
    for name, in_f, out_f in [
        ('linear_layers', num_features * num_networks, num_features),
        ('FF_layer1', num_features, d_ff),
        ('FF_layer2', d_ff, num_features),
    ]:
        for g in range(num_genes):
            sd[f'GATblock.{name}.{g}.weight'] = torch.randn(out_f, in_f)
            sd[f'GATblock.{name}.{g}.bias'] = torch.randn(out_f)

    # GAT_layers flat nn.Sequential of GATv2_onehop (just a couple of dummy conv1 params)
    for flat_idx in range(num_GATblock * num_networks):
        sd[f'GATblock.GAT_layers.{flat_idx}.conv1.bias'] = torch.randn(num_features)

    # SSL head per-gene linear (final_linear1[ssl_task_index] is a ModuleList of Linears)
    for g in range(num_genes):
        sd[f'final_linear1.{ssl_task_index}.{g}.weight'] = torch.randn(output_sizes[ssl_task_index], num_features)
        sd[f'final_linear1.{ssl_task_index}.{g}.bias'] = torch.randn(output_sizes[ssl_task_index])

    # a plain (non-per-gene) final_linear1 entry, passthrough untouched
    sd['final_linear1.1.weight'] = torch.randn(output_sizes[1], num_features)
    sd['final_linear1.1.bias'] = torch.randn(output_sizes[1])

    return sd


def test_adapter_stacks_per_gene_keys_and_reindexes_gat_layers(tiny_model_config):
    cfg = tiny_model_config
    network_edges = make_network_edges(cfg['num_genes'], cfg['num_networks'])
    model = MutationProjector(**tiny_model_kwargs(cfg, network_edges))

    legacy = build_legacy_state_dict(
        cfg['num_genes'], cfg['num_features'], cfg['num_networks'], cfg['num_GATblock'],
        cfg['d_ff'], cfg['output_sizes'], cfg['ssl_task_index'],
    )
    adapted = adapt_legacy_state_dict(model, legacy)

    # stacked shape correctness
    assert adapted['GATblock.linear_layers.weight'].shape == (cfg['num_genes'], cfg['num_features'], cfg['num_features'] * cfg['num_networks'])
    assert adapted['GATblock.linear_layers.bias'].shape == (cfg['num_genes'], cfg['num_features'])

    # per-gene slice equality (no data corruption/misalignment during stacking)
    for g in range(cfg['num_genes']):
        assert torch.equal(adapted['GATblock.linear_layers.weight'][g], legacy[f'GATblock.linear_layers.{g}.weight'])
        assert torch.equal(adapted['GATblock.FF_layer1.bias'][g], legacy[f'GATblock.FF_layer1.{g}.bias'])
        assert torch.equal(adapted[f'final_linear1.{cfg["ssl_task_index"]}.weight'][g], legacy[f'final_linear1.{cfg["ssl_task_index"]}.{g}.weight'])

    # GAT_layers flat -> nested reindexing: idx = i*num_networks+j
    for i in range(cfg['num_GATblock']):
        for j in range(cfg['num_networks']):
            flat_idx = i * cfg['num_networks'] + j
            assert torch.equal(
                adapted[f'GATblock.GAT_layers.{i}.{j}.conv1.bias'],
                legacy[f'GATblock.GAT_layers.{flat_idx}.conv1.bias'],
            )

    # passthrough key untouched
    assert torch.equal(adapted['final_linear1.1.weight'], legacy['final_linear1.1.weight'])


def test_adapter_passes_new_format_keys_through(tiny_model_config):
    cfg = tiny_model_config
    kwargs = tiny_model_kwargs(cfg, make_network_edges(cfg['num_genes'], cfg['num_networks']))
    new_format = MutationProjector(**kwargs).state_dict()
    target = MutationProjector(**kwargs)
    target.load_state_dict(adapt_legacy_state_dict(target, new_format), strict=True)
