"""
Benchmark the real pretrained_model.pth forward pass:
  1. old (pre-refactor) vs. new (refactored) code, both on CPU -- the algorithmic speedup
     from vectorizing the per-gene loops / removing PyG Data-DataLoader construction / etc.
  2. new code across devices (cpu, mps, cuda) -- the additional speedup from Apple Silicon's
     MPS backend (GPU / unified memory) or CUDA, on top of the refactor.

The old (pre-refactor) implementation hardcodes `.cuda(device)` everywhere and can only
target actual CUDA devices -- calling it on a Mac raises immediately, even before you get to
compare speed. To include it in this benchmark at all, `.cuda()` is monkeypatched to a no-op
for the duration of the old-model benchmark only (same technique used in tests/conftest.py) so
it runs on CPU tensors. This changes nothing about what the old code computes, only lets it run
without a real CUDA device present. The new (refactored) code needs no such shim -- it uses
`.to(device)` and genuinely runs on cpu/mps/cuda.

Usage (run from inside src/, matching this repo's existing scripts' path conventions):
    python benchmark_device.py
    python benchmark_device.py --devices cpu mps --batch-size 64 --repeats 10
    python benchmark_device.py --skip-old-baseline --devices mps
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from load_model import adapt_legacy_state_dict
from MutationProjector_nn import MutationProjector as NewMutationProjector
from nn_training_functions import merge_data, convert_mutations

FI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FI_DIR / 'tests'))  # OldMutationProjector is the frozen pre-refactor baseline


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


def _production_hparams(network_edges, device):
    import numpy as np
    return dict(
        num_genes=468, num_features=10, network_edges=network_edges, num_GATblock=2, num_heads=1,
        dropout_p=0.1, cuda_device=device, output_sizes=[3, 10, 3], mask_percentage=0, input_genes=[],
        d_ff=10, use_representative_embedding=1, ssl_task_index=0, use_special_token=1,
        num_special_tokens=9, num_bins=list(np.append([5, 5], [2] * 7)), use_pooling=0,
    )


def benchmark_old_baseline(network_edges, checkpoint, X_converted, X_special, n_repeats):
    from reference_impls import OldMutationProjector

    # Shim only: lets the CUDA-hardcoded old code run on CPU tensors on a machine with no CUDA
    # device. Does not change what it computes -- .cuda(x) just becomes a no-op returning self.
    orig_tensor_cuda = torch.Tensor.cuda
    orig_module_cuda = torch.nn.Module.cuda
    torch.Tensor.cuda = lambda self, *a, **kw: self
    torch.nn.Module.cuda = lambda self, *a, **kw: self
    try:
        print("\n=== old (pre-refactor) code, cpu, via .cuda()-shim ===")
        hp = _production_hparams(network_edges, device=0)  # old code only understands int cuda indices
        model = OldMutationProjector(**hp)
        model.load_state_dict(checkpoint, strict=True)  # old checkpoint format IS the native old-model format
        model.eval()
        t = time_forward(model, X_converted, X_special, torch.device('cpu'), n_repeats=n_repeats)
        print(f"  {t * 1000:.1f} ms/call (avg of {n_repeats} calls)")
        return t
    finally:
        torch.Tensor.cuda = orig_tensor_cuda
        torch.nn.Module.cuda = orig_module_cuda


def benchmark_new(network_edges, checkpoint, X_converted, X_special, device_str, n_repeats):
    device = torch.device(device_str)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        print(f"\nskipping new/{device_str}: MPS not available (needs Apple Silicon + a PyTorch build with MPS support)")
        return None
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f"\nskipping new/{device_str}: CUDA not available")
        return None

    print(f"\n=== new (refactored) code, device: {device_str} ===")
    hp = _production_hparams(network_edges, device=device_str)
    model = NewMutationProjector(**hp)
    model.load_state_dict(adapt_legacy_state_dict(model, checkpoint), strict=True)
    model.eval()
    sample_param = next(model.parameters())
    print(f"  confirmed model parameters live on: {sample_param.device}")

    t = time_forward(model, X_converted.to(device), X_special.to(device), device, n_repeats=n_repeats)
    print(f"  {t * 1000:.1f} ms/call (avg of {n_repeats} calls)")
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--devices', nargs='+', default=['cpu', 'mps'],
                         help="new-code devices to benchmark, e.g. cpu mps cuda:0")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--skip-old-baseline', action='store_true',
                         help="skip the pre-refactor code comparison, only benchmark new code across --devices")
    args = parser.parse_args()

    networks = 'GRN_expanded;E3_expanded;phosphorylation_expanded;physical_ppi_expanded;genetic_interaction_expanded;DDRAM;STRING;PCNET'.split(';')
    network_edges = [torch.load(FI_DIR / 'data' / 'networks' / f'{n}.pt', map_location='cpu') for n in networks]
    checkpoint = torch.load(FI_DIR / 'pretrained_model' / 'pretrained_model.pth', map_location='cpu')

    path_data = FI_DIR / 'data' / 'downstream_data' / 'eval_dataset' / 'test_sample'
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
    if not args.skip_old_baseline:
        results['old/cpu'] = benchmark_old_baseline(network_edges, checkpoint, X_converted, X_special, args.repeats)

    for device_str in args.devices:
        t = benchmark_new(network_edges, checkpoint, X_converted, X_special, device_str, args.repeats)
        if t is not None:
            results[f'new/{device_str}'] = t

    print(f"\n=== summary (batch_size={args.batch_size}) ===")
    for name, t in results.items():
        print(f"  {name:12s} {t * 1000:8.1f} ms/call")

    if 'old/cpu' in results and 'new/cpu' in results:
        print(f"\nrefactor speedup (new/cpu vs old/cpu): {results['old/cpu'] / results['new/cpu']:.2f}x")
    if 'new/cpu' in results:
        for name, t in results.items():
            if name not in ('old/cpu', 'new/cpu'):
                print(f"device speedup ({name} vs new/cpu): {results['new/cpu'] / t:.2f}x")


if __name__ == '__main__':
    main()
