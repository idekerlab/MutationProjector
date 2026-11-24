"""Run B (improved recipe): embed a converted MP2 cohort with global bin edges and per-sample covariate padding."""
import argparse
from pathlib import Path

import numpy as np
import torch

from generate_embeddings import embed_cohort
from load_model import adapt_legacy_state_dict, load_network_edges, production_hparams
from mp2_cohort import NUM_BINS, load_manifest
from MutationProjector_nn import MutationProjector
from nn_training_functions import bin_edges

FI_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', required=True, help='inputs/ directory written by convert_mp2_cohort.py')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    inputs = Path(args.inputs_dir)
    manifest = load_manifest(inputs)
    cov_missing = np.load(inputs / 'cov_missing.npy')
    # A covariate with no observed values (Clock_SBS1/SBS5) has no edges. Its rows are all
    # missing and get the zero token, so any edges work.
    edges = [bin_edges(0.0, 0.0, k) if e is None else np.asarray(e) for e, k in zip(manifest['global_bin_edges'], NUM_BINS)]

    model = MutationProjector(**production_hparams(load_network_edges(FI_DIR), args.device))
    checkpoint = torch.load(FI_DIR / 'pretrained_model' / 'pretrained_model.pth', map_location='cpu')
    model.load_state_dict(adapt_legacy_state_dict(model, checkpoint), strict=True)

    embed_cohort(model, np.load(inputs / 'X_converted.npy'), np.load(inputs / 'X_special.npy'), cov_missing,
                 edges, args.out_dir, batch_size=args.batch_size)
    print(f"wrote {manifest['n_samples']} rows to {args.out_dir}")


if __name__ == '__main__':
    main()
