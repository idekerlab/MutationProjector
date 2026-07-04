import random

import torch

from nn_training_functions import tokenizer, convert_mutations
from reference_impls import OldTokenizer


def _build_pair(seed, num_genes, num_features, num_gene_features):
    torch.manual_seed(seed)
    tok_new = tokenizer(num_genes, num_features, cuda_device=0, num_gene_features=num_gene_features)
    torch.manual_seed(seed)
    tok_old = OldTokenizer(num_genes, num_features, cuda_device=0, num_gene_features=num_gene_features)
    # same seed + same construction order -> identical embedding weights for a fair comparison
    assert torch.equal(tok_new.mut_embedding.weight, tok_old.mut_embedding.weight)
    assert torch.equal(tok_new.gene_embedding.weight, tok_old.gene_embedding.weight)
    return tok_new, tok_old


def test_matches_reference_loop_no_masking():
    for trial in range(15):
        rng = random.Random(trial)
        num_genes = rng.randint(3, 25)
        num_features = rng.randint(2, 8)
        num_gene_features = 3
        B = rng.randint(1, 6)

        tok_new, tok_old = _build_pair(trial, num_genes, num_features, num_gene_features)

        X_raw = torch.randint(0, 2, (B, num_genes, num_gene_features))
        X_converted = convert_mutations(X_raw)

        out_new, pos_new = tok_new.return_mut_embedding(X_converted.clone(), 0, False)
        out_old, pos_old = tok_old.return_mut_embedding(X_converted.clone(), 0, False)

        assert pos_new == pos_old == []
        assert torch.allclose(out_old, out_new, atol=1e-6), f"trial={trial}"


def test_matches_reference_loop_with_explicit_masking():
    # test_geneset accepts an explicit list of gene indices to mask (deterministic, no RNG),
    # so both old and new see identical masked positions.
    for trial in range(15):
        rng = random.Random(trial + 500)
        num_genes = rng.randint(5, 25)
        num_features = rng.randint(2, 8)
        num_gene_features = 3
        B = rng.randint(1, 6)
        num_masked = rng.randint(0, num_genes)
        positions_to_mask = sorted(rng.sample(range(num_genes), num_masked))

        tok_new, tok_old = _build_pair(trial, num_genes, num_features, num_gene_features)

        X_raw = torch.randint(0, 2, (B, num_genes, num_gene_features))
        X_converted = convert_mutations(X_raw)

        out_new, pos_new = tok_new.return_mut_embedding(X_converted.clone(), 0, positions_to_mask)
        out_old, pos_old = tok_old.return_mut_embedding(X_converted.clone(), 0, positions_to_mask)

        assert pos_new == pos_old == positions_to_mask
        assert torch.allclose(out_old, out_new, atol=1e-6), f"trial={trial}"


def test_all_genes_masked_edge_case():
    num_genes, num_features, num_gene_features, B = 6, 3, 3, 4
    tok_new, tok_old = _build_pair(0, num_genes, num_features, num_gene_features)
    X_raw = torch.randint(0, 2, (B, num_genes, num_gene_features))
    X_converted = convert_mutations(X_raw)
    positions_to_mask = list(range(num_genes))

    out_new, _ = tok_new.return_mut_embedding(X_converted.clone(), 0, positions_to_mask)
    out_old, _ = tok_old.return_mut_embedding(X_converted.clone(), 0, positions_to_mask)
    assert torch.allclose(out_old, out_new, atol=1e-6)
