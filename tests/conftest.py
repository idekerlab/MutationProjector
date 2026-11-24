import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """tests/reference_impls.py calls .cuda() everywhere. Make it a no-op so the old code runs on CPU."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **kw: self)


@pytest.fixture(autouse=True)
def seeded_rng():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def tiny_model_config():
    return dict(
        num_genes=9,
        num_features=4,
        num_networks=3,
        num_GATblock=2,
        num_heads=1,
        d_ff=5,
        num_special_tokens=2,
        num_bins=[3, 4],
        output_sizes=[2, 3, 2],
        ssl_task_index=0,
        cuda_device='cpu',
        dropout_p=0.0,
    )


def tiny_model_kwargs(cfg, network_edges, **overrides):
    """MutationProjector keyword arguments for tiny_model_config. Overrides replace defaults."""
    kwargs = dict(
        num_genes=cfg['num_genes'], num_features=cfg['num_features'], network_edges=network_edges,
        num_GATblock=cfg['num_GATblock'], num_heads=cfg['num_heads'], dropout_p=cfg['dropout_p'],
        cuda_device=cfg['cuda_device'], output_sizes=cfg['output_sizes'], mask_percentage=0, input_genes=[],
        d_ff=cfg['d_ff'], ssl_task_index=cfg['ssl_task_index'], use_representative_embedding=True,
        use_special_token=True, num_special_tokens=cfg['num_special_tokens'], num_bins=cfg['num_bins'],
        use_pooling=True,
    )
    kwargs.update(overrides)
    return kwargs


def seeded_pair(seed, new_cls, old_cls, *args, **kwargs):
    """Builds new_cls and old_cls from the same seed, so both get the same initial weights."""
    torch.manual_seed(seed)
    new = new_cls(*args, **kwargs)
    torch.manual_seed(seed)
    old = old_cls(*args, **kwargs)
    return new, old


def make_network_edges(num_genes, num_networks, generator=None):
    edges = []
    for _ in range(num_networks):
        src = torch.randint(0, num_genes, (num_genes,), generator=generator)
        dst = torch.randint(0, num_genes, (num_genes,), generator=generator)
        edges.append(torch.stack([src, dst], dim=0))
    return edges
