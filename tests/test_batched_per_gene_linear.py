"""
Item 8 (core numerics): BatchedPerGeneLinear (one torch.einsum) vs. a reference
nn.ModuleList([nn.Linear(...) for _ in range(num_genes)]) applied via a per-gene Python loop,
given identical per-gene weights.
"""
import torch
import torch.nn as nn

from GATv2_functions import BatchedPerGeneLinear


def loop_apply(module_list, x):
    out = [module_list[g](x[:, g, :]) for g in range(len(module_list))]
    return torch.stack(out, dim=1)


def test_matches_looped_modulelist_of_linear():
    torch.manual_seed(0)
    for trial in range(10):
        num_genes = torch.randint(1, 30, (1,)).item()
        in_f = torch.randint(1, 12, (1,)).item()
        out_f = torch.randint(1, 12, (1,)).item()
        batch = torch.randint(1, 8, (1,)).item()

        ref = nn.ModuleList([nn.Linear(in_f, out_f) for _ in range(num_genes)])
        batched = BatchedPerGeneLinear(num_genes, in_f, out_f)
        batched.weight.data.copy_(torch.stack([ref[g].weight for g in range(num_genes)]))
        batched.bias.data.copy_(torch.stack([ref[g].bias for g in range(num_genes)]))

        x = torch.randn(batch, num_genes, in_f)
        with torch.no_grad():
            out_ref = loop_apply(ref, x)
            out_batched = batched(x)

        assert torch.allclose(out_ref, out_batched, atol=1e-5), f"trial={trial}"


def test_distinct_per_gene_weights_produce_distinct_outputs():
    # sanity: confirms the batched op doesn't accidentally share weights across genes
    torch.manual_seed(0)
    num_genes, in_f, out_f, batch = 5, 3, 4, 2
    batched = BatchedPerGeneLinear(num_genes, in_f, out_f)
    x = torch.ones(batch, num_genes, in_f)
    out = batched(x)
    # with random distinct weights, genes should (almost surely) not all produce equal output
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])
