import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """
    Production code hardcodes `.cuda(self.cuda_device)` throughout (GATv2block,
    MutationProjector, tokenizer, tokenize_special_tokens, ...). CI/dev machines running this
    suite may have no GPU at all, and `.cuda()` raises unconditionally in that case. Monkeypatch
    it to a no-op for the duration of each test so the exact same production code paths run on
    CPU tensors -- this changes nothing about *what* is computed, only *where*.
    """
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **kw: self)


@pytest.fixture(autouse=True)
def seeded_rng():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def tiny_model_config():
    """Small synthetic dims so structural/equivalence tests run in milliseconds without the
    real 468-gene checkpoint."""
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
        cuda_device=0,
        dropout_p=0.0,
    )


def make_network_edges(num_genes, num_networks, generator=None):
    edges = []
    for _ in range(num_networks):
        src = torch.randint(0, num_genes, (num_genes,), generator=generator)
        dst = torch.randint(0, num_genes, (num_genes,), generator=generator)
        edges.append(torch.stack([src, dst], dim=0))
    return edges


def assert_allclose(old, new, rtol=1e-4, atol=1e-6, msg=""):
    __tracebackhide__ = True
    if not torch.allclose(old, new, rtol=rtol, atol=atol):
        max_diff = (old - new).abs().max().item()
        raise AssertionError(f"{msg} allclose failed: max diff = {max_diff:.3e} (rtol={rtol}, atol={atol})")
