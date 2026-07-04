"""
Benchmark the real pretrained_model.pth forward pass across devices (cpu, mps, cuda) to
measure the effect of both the training-code refactor and Apple Silicon's MPS backend
(GPU / unified memory) vs. CPU.

Usage (run from inside src/, matching this repo's existing scripts' path conventions):
    python benchmark_device.py
    python benchmark_device.py --devices cpu mps --batch-size 64 --repeats 10
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from load_model import load_MutationProjector
from nn_training_functions import merge_data, convert_mutations


def time_forward(model, X_converted, X_special, device, n_repeats=5):
    model.eval()
    with torch.no_grad():
        # warmup (first call on a new device includes kernel compilation/allocation overhead)
        model(X_converted, X_special_tokens=X_special, test_geneset=False,
              return_attention_weights=False, apply_paddings=False)
        _sync(device)
        t0 = time.time()
        for _ in range(n_repeats):
            model(X_converted, X_special_tokens=X_special, test_geneset=False,
                  return_attention_weights=False, apply_paddings=False)
        _sync(device)
        return (time.time() - t0) / n_repeats


def _sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', default=['cpu', 'mps'],
                         help="devices to benchmark, e.g. cpu mps cuda:0")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()

    fi_dir = Path(__file__).resolve().parents[1]
    path_data = fi_dir / 'data' / 'downstream_data' / 'eval_dataset' / 'test_sample'
    mdf = pd.read_csv(path_data / 'mut.txt', sep='\t')
    cna = pd.read_csv(path_data / 'cna.txt', sep='\t')
    cnd = pd.read_csv(path_data / 'cnd.txt', sep='\t')
    X_test, _, _ = merge_data(mdf, cna, cnd, use_cancer_types=False)
    covariates = pd.read_csv(path_data / 'covariates.txt', sep='\t')
    X_special_full = torch.tensor(covariates.set_index('sample').values)
    X_converted_full = convert_mutations(X_test)

    # tile the (small, 20-sample) shipped test set up to --batch-size for a more realistic load
    reps = (args.batch_size + X_converted_full.shape[0] - 1) // X_converted_full.shape[0]
    X_converted = X_converted_full.repeat(reps, 1)[:args.batch_size]
    X_special = X_special_full.repeat(reps, 1)[:args.batch_size]

    results = {}
    for device_str in args.devices:
        device = torch.device(device_str)
        if device.type == 'mps' and not torch.backends.mps.is_available():
            print(f"skipping {device_str}: MPS not available (needs Apple Silicon + a PyTorch build with MPS support)")
            continue
        if device.type == 'cuda' and not torch.cuda.is_available():
            print(f"skipping {device_str}: CUDA not available")
            continue

        print(f"\n=== device: {device_str} ===")
        model = load_MutationProjector(device=device_str)
        sample_param = next(model.parameters())
        print(f"  confirmed model parameters live on: {sample_param.device}")

        X_converted_dev = X_converted.to(device)
        X_special_dev = X_special.to(device)
        t = time_forward(model, X_converted_dev, X_special_dev, device, n_repeats=args.repeats)
        results[device_str] = t
        print(f"  {t*1000:.1f} ms/call (batch_size={args.batch_size}, avg of {args.repeats} calls)")

    if 'cpu' in results:
        for device_str, t in results.items():
            if device_str != 'cpu':
                print(f"\nSpeedup {device_str} vs cpu: {results['cpu']/t:.2f}x")


if __name__ == '__main__':
    main()
