"""MP2 cohort conversion, fixed bin edges and per-sample covariate padding."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from generate_embeddings import embed_cohort, embed_from_pretrained
from load_geneList import load_genes
from load_model import adapt_legacy_state_dict, load_eval_inputs, load_network_edges, production_hparams
from mp2_cohort import COVARIATES, build_alterations, convert_cohort, map_genes
from MutationProjector_nn import MutationProjector
from nn_training_functions import bin_edges, convert_mutations, merge_data, tokenize_special_tokens

FI_DIR = Path(__file__).resolve().parents[1]


def test_fixed_edges_equal_to_batch_edges_reproduce_default():
    tok = tokenize_special_tokens(num_features=3, num_bins=5, cuda_device='cpu')
    values = torch.randn(50, dtype=torch.float64) * 10
    fixed = tok(values, edges=bin_edges(values.min().item(), values.max().item(), 5))
    assert torch.equal(tok(values), fixed)


def test_missing_rows_get_the_padding_zero_token():
    tok = tokenize_special_tokens(num_features=3, num_bins=5, cuda_device='cpu')
    values = torch.tensor([0.0, float('nan'), 2.0, 3.0], dtype=torch.float64)
    missing = torch.tensor([False, True, False, True])
    edges = bin_edges(0.0, 3.0, 5)
    out = tok(values, edges=edges, missing=missing)
    assert torch.equal(out[missing], torch.zeros(2, 3))
    assert torch.equal(out[~missing], tok(values[~missing], edges=edges))
    all_missing = tok(values, edges=edges, missing=torch.ones(4, dtype=torch.bool))
    assert torch.equal(all_missing, tok(values, apply_padding=True))


def test_map_genes_prefers_exact_name_then_unique_alias():
    assert map_genes(['WWTR1'], ['TAZ', 'WWTR1']) == ['WWTR1']
    assert map_genes(['WWTR1'], ['TAZ']) == ['TAZ']
    with pytest.raises(ValueError):
        map_genes(['NOT_A_GENE'], ['TP53'])


def test_build_alterations_thresholds_and_profiled_masks():
    y_mut = np.array([[1.0, 0.0, np.nan]])
    x_cnv = np.array([[2.0, -1.5, np.nan]])
    X, mut_profiled, cna_profiled = build_alterations(y_mut, x_cnv)
    assert X.tolist() == [[[1, 1, 0], [0, 0, 0], [0, 0, 0]]]
    assert mut_profiled.tolist() == [[True, True, False]]
    assert cna_profiled.tolist() == [[True, True, False]]


def test_run_a_text_files_and_run_b_arrays_hold_the_same_rows(tmp_path):
    genes = load_genes('MSKIMPACT468')
    rng = np.random.default_rng(0)
    n = 7
    src = tmp_path / 'cohort'
    src.mkdir()
    np.save(src / 'samples.npy', np.array([f'S{i:02d}' for i in range(n)]))
    np.save(src / 'genes.npy', np.array(genes))
    mut = rng.choice(['0.0', '1.0', None], size=(n, len(genes)))
    cnv = rng.choice(['0.0', '2.0', '-2.0', '-1.0', None], size=(n, len(genes)))
    pd.DataFrame(mut, columns=genes).to_parquet(src / 'y_mut.parquet')
    pd.DataFrame(cnv, columns=genes).to_parquet(src / 'X_cnv.parquet')
    cov = pd.DataFrame({'TMB': ['LOW', None, 'HIGH', 'VERY HIGH', 'LOW', 'MODERATE', None],
                        'aneuploidy': ['NO', 'LOW', None, 'VERY HIGH', 'HIGH', 'MODERATE', 'NO']})
    for col in ['APOBEC', 'MMR_deficiency', 'POL_deficiency', 'Tobacco', 'UV']:
        cov[f'{col}_signatures'] = rng.choice(['0.0', '1.0', None], size=n)
    cov.to_parquet(src / 'X_genomic_cov.parquet')

    manifest = convert_cohort(src, tmp_path / 'out')
    run_a, inputs = tmp_path / 'out' / 'run_a_repo', tmp_path / 'out' / 'inputs'
    frames = [pd.read_csv(run_a / f'{k}.txt', sep='\t') for k in ('mut', 'cna', 'cnd')]
    X_a, _, _ = merge_data(*frames, use_cancer_types=False)
    covariates = pd.read_csv(run_a / 'covariates.txt', sep='\t')

    assert manifest['n_samples'] == n
    assert torch.equal(convert_mutations(X_a), torch.from_numpy(np.load(inputs / 'X_converted.npy')).long())
    assert covariates['sample'].tolist() == np.load(inputs / 'samples.npy').tolist()
    assert list(covariates.columns[1:]) == COVARIATES
    X_special = np.load(inputs / 'X_special.npy')
    assert np.array_equal(np.nan_to_num(X_special), covariates[COVARIATES].to_numpy(dtype=np.float32))
    assert np.array_equal(np.load(inputs / 'cov_missing.npy'), np.isnan(X_special))


@pytest.mark.slow
def test_run_b_matches_run_a_on_test_sample(tmp_path, monkeypatch):
    """With edges from the batch itself and no missing values, Run B must reproduce Run A."""
    sample_dir = tmp_path / 'test_sample'
    shutil.copytree(FI_DIR / 'data' / 'downstream_data' / 'eval_dataset' / 'test_sample', sample_dir)
    monkeypatch.chdir(FI_DIR / 'src')
    embed_from_pretrained('pretrained_model.pth', dataset='test_sample', dataset_type='eval_dataset',
                          cuda_device='cpu', path_dataset=str(sample_dir))

    X_converted, X_special = load_eval_inputs(sample_dir)
    edges = [bin_edges(X_special[:, j].min().item(), X_special[:, j].max().item(), k)
             for j, k in enumerate(production_hparams([], 'cpu')['num_bins'])]
    model = MutationProjector(**production_hparams(load_network_edges(FI_DIR), 'cpu'))
    checkpoint = torch.load(FI_DIR / 'pretrained_model' / 'pretrained_model.pth', map_location='cpu')
    model.load_state_dict(adapt_legacy_state_dict(model, checkpoint), strict=True)
    embed_cohort(model, X_converted.numpy(), X_special.numpy(), np.zeros(X_special.shape, dtype=bool), edges,
                 tmp_path / 'run_b', batch_size=256)

    for name in ('gene_emb', 'cov_emb', 'rep_emb', 'final_layer_emb'):
        run_a = torch.load(sample_dir / 'prediction_results' / f'{name}.pt', map_location='cpu').numpy()
        run_b = np.load(tmp_path / 'run_b' / f'{name}.npy')
        assert np.allclose(run_a, run_b, atol=1e-5), name
