"""
Convert an MP2 processed cohort (for example merged_v001) into MP1 inputs.

Source directory: samples.npy, genes.npy, y_mut.parquet, X_cnv.parquet, X_genomic_cov.parquet.
Parquet cells are strings ('0.0', '1.0', ...) or NA. Rows follow samples.npy.

Outputs, rows in sorted sample order:
- Run A (repo recipe): mut.txt, cna.txt, cnd.txt, covariates.txt for embed_from_pretrained.
  Covariate NA is written as 0, because the repo recipe has no per-sample missingness.
- Run B (improved recipe): X_converted.npy, X_special.npy (NaN kept), cov_missing.npy.
- Both: samples.npy, mut_profiled.npy, cna_profiled.npy, full_coverage_idx.npy, manifest.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from load_geneList import convert_geneIDs, load_genes
from nn_training_functions import bin_edges, convert_mutations

SCHEMA_VERSION = 1

# MP1 covariate order and bin counts (pretrained_model.pth)
COVARIATES = ['TMB', 'aSCNA', 'APOBEC', 'Clock_SBS1', 'Clock_SBS5', 'MMR', 'POLE', 'Tobacco', 'UV']
NUM_BINS = [5, 5, 2, 2, 2, 2, 2, 2, 2]

BINARY = {'0.0': 0.0, '1.0': 1.0}
# MP1 covariate -> (MP2 column, category -> ordinal value). None: no MP2 source.
COVARIATE_SOURCES = {
    'TMB': ('TMB', {'LOW': 0.0, 'MODERATE': 1.0, 'HIGH': 2.0, 'VERY HIGH': 3.0}),
    'aSCNA': ('aneuploidy', {'NO': 0.0, 'LOW': 1.0, 'MODERATE': 2.0, 'HIGH': 3.0, 'VERY HIGH': 4.0}),
    'APOBEC': ('APOBEC_signatures', BINARY),
    'Clock_SBS1': None,
    'Clock_SBS5': None,
    'MMR': ('MMR_deficiency_signatures', BINARY),
    'POLE': ('POL_deficiency_signatures', BINARY),
    'Tobacco': ('Tobacco_signatures', BINARY),
    'UV': ('UV_signatures', BINARY),
}

# GISTIC calls. Only deep changes count as altered (assumption: MP1 does not define its cna/cnd thresholds).
CNA_AMP = 2.0
CNA_DEL = -2.0
MUT_VALUES = {0.0, 1.0}
CNV_VALUES = {-2.0, -1.5, -1.0, 0.0, 1.0, 2.0}


def map_genes(mp1_genes, mp2_genes):
    """
    Returns the MP2 column name for each MP1 gene, in mp1_genes order.
    Uses the exact name when MP2 has it. Otherwise uses the one MP2 name whose primary ID
    (convert_geneIDs) equals the MP1 gene. Raises ValueError if a gene has no match or
    more than one alias match.
    """
    mp2_set = set(mp2_genes)
    by_primary = {}
    for raw, primary in zip(mp2_genes, convert_geneIDs(list(mp2_genes))):
        by_primary.setdefault(primary, []).append(raw)
    out, problems = [], []
    for gene in mp1_genes:
        if gene in mp2_set:
            out.append(gene)
        elif len(by_primary.get(gene, [])) == 1:
            out.append(by_primary[gene][0])
        else:
            problems.append((gene, by_primary.get(gene, [])))
    if problems:
        raise ValueError(f"unresolved MP1 genes (gene, MP2 candidates): {problems}")
    return out


def read_numeric_matrix(path, columns, allowed):
    """Reads the given parquet columns as an (N, len(columns)) float32 array, NA as NaN. Raises on values outside allowed."""
    frame = pq.read_table(path, columns=list(columns)).to_pandas()
    values = frame.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    seen = set(np.unique(values[~np.isnan(values)]).tolist())
    if not seen <= allowed:
        raise ValueError(f"{path}: unexpected values {sorted(seen - allowed)}")
    return values


def build_alterations(y_mut, x_cnv):
    """
    y_mut, x_cnv: (N, G) float arrays, NaN for not profiled.
    Returns X (N, G, 3) uint8 [mut, cna, cnd] with NaN read as not altered,
    mut_profiled (N, G) bool and cna_profiled (N, G) bool.
    """
    mut = y_mut == 1.0
    cna = x_cnv == CNA_AMP
    cnd = x_cnv == CNA_DEL
    X = np.stack([mut, cna, cnd], axis=-1).astype(np.uint8)
    return X, ~np.isnan(y_mut), ~np.isnan(x_cnv)


def build_covariates(cov_frame):
    """
    cov_frame: MP2 X_genomic_cov as a DataFrame of strings/NA.
    Returns values (N, 9) float32 in COVARIATES order with NaN for missing, and missing (N, 9) bool.
    Raises ValueError on a category that has no mapping.
    """
    values = np.full((len(cov_frame), len(COVARIATES)), np.nan, dtype=np.float32)
    for j, name in enumerate(COVARIATES):
        source = COVARIATE_SOURCES[name]
        if source is None:
            continue
        column, mapping = source
        raw = cov_frame[column]
        unknown = set(raw.dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"{column}: unmapped categories {sorted(unknown)}")
        values[:, j] = raw.map(mapping).to_numpy(dtype=np.float32, na_value=np.nan)
    return values, np.isnan(values)


def global_bin_edges(values, missing):
    """Returns one (num_bins+1,) edge array per covariate, from the cohort's non-missing values. A covariate with no values gets None."""
    edges = []
    for j, num_bins in enumerate(NUM_BINS):
        observed = values[~missing[:, j], j].astype(np.float64)
        edges.append(None if observed.size == 0 else bin_edges(observed.min(), observed.max(), num_bins))
    return edges


def write_gene_table(path, samples, genes, matrix):
    frame = pd.DataFrame(matrix, columns=genes)
    frame.insert(0, 'sample', samples)
    frame.to_csv(path, sep='\t', index=False)


def convert_cohort(cohort_dir, out_dir, limit=None):
    """
    Converts one MP2 cohort directory. Writes Run A files to out_dir/run_a_repo/ and shared and
    Run B files to out_dir/inputs/. limit keeps the first limit rows (for a pilot).
    Returns the manifest dict.
    """
    cohort_dir, out_dir = Path(cohort_dir), Path(out_dir)
    samples = np.load(cohort_dir / 'samples.npy', allow_pickle=True).astype(str)
    if not (samples[:-1] < samples[1:]).all():
        raise ValueError("samples.npy must be strictly sorted, so the row order matches merge_data's sorted order")
    mp1_genes = load_genes('MSKIMPACT468')
    columns = map_genes(mp1_genes, list(np.load(cohort_dir / 'genes.npy', allow_pickle=True).astype(str)))

    y_mut = read_numeric_matrix(cohort_dir / 'y_mut.parquet', columns, MUT_VALUES)
    x_cnv = read_numeric_matrix(cohort_dir / 'X_cnv.parquet', columns, CNV_VALUES)
    cov_frame = pq.read_table(cohort_dir / 'X_genomic_cov.parquet').to_pandas()
    if not (len(y_mut) == len(x_cnv) == len(cov_frame) == len(samples)):
        raise ValueError("row counts differ between samples.npy and the parquet files")
    if limit is not None:
        samples, y_mut, x_cnv, cov_frame = samples[:limit], y_mut[:limit], x_cnv[:limit], cov_frame.iloc[:limit]

    X, mut_profiled, cna_profiled = build_alterations(y_mut, x_cnv)
    cov_values, cov_missing = build_covariates(cov_frame)
    edges = global_bin_edges(cov_values, cov_missing)
    full_coverage_idx = np.flatnonzero(mut_profiled.all(axis=1) & cna_profiled.all(axis=1))

    inputs_dir, run_a_dir = out_dir / 'inputs', out_dir / 'run_a_repo'
    inputs_dir.mkdir(parents=True, exist_ok=True)
    run_a_dir.mkdir(parents=True, exist_ok=True)

    np.save(inputs_dir / 'samples.npy', samples)
    np.save(inputs_dir / 'X_converted.npy', convert_mutations(torch.from_numpy(X)).numpy().astype(np.uint8))
    np.save(inputs_dir / 'X_special.npy', cov_values)
    np.save(inputs_dir / 'cov_missing.npy', cov_missing)
    np.save(inputs_dir / 'mut_profiled.npy', mut_profiled)
    np.save(inputs_dir / 'cna_profiled.npy', cna_profiled)
    np.save(inputs_dir / 'full_coverage_idx.npy', full_coverage_idx)

    for name, k in (('mut', 0), ('cna', 1), ('cnd', 2)):
        write_gene_table(run_a_dir / f'{name}.txt', samples, mp1_genes, X[:, :, k])
    covariates = pd.DataFrame(np.nan_to_num(cov_values, nan=0.0), columns=COVARIATES)
    covariates.insert(0, 'sample', samples)
    covariates.to_csv(run_a_dir / 'covariates.txt', sep='\t', index=False)

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'source_dir': str(cohort_dir),
        'limit': limit,
        'n_samples': int(len(samples)),
        'n_full_coverage': int(len(full_coverage_idx)),
        'genes_mp1': mp1_genes,
        'genes_mp2_columns': columns,
        'cna_threshold': {'cna': CNA_AMP, 'cnd': CNA_DEL},
        'covariates': COVARIATES,
        'num_bins': NUM_BINS,
        'covariate_sources': {k: (None if v is None else {'column': v[0], 'map': v[1]}) for k, v in COVARIATE_SOURCES.items()},
        'covariate_missing_counts': dict(zip(COVARIATES, cov_missing.sum(axis=0).astype(int).tolist())),
        'global_bin_edges': [None if e is None else e.tolist() for e in edges],
        'mut_profiled_fraction': float(mut_profiled.mean()),
        'cna_profiled_fraction': float(cna_profiled.mean()),
    }
    (inputs_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def load_manifest(inputs_dir):
    """Reads manifest.json and checks the schema version."""
    manifest = json.loads((Path(inputs_dir) / 'manifest.json').read_text())
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version {manifest.get('schema_version')} != {SCHEMA_VERSION}")
    return manifest
