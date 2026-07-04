"""
Benchmark the real pretrained_model.pth:
  1. old (pre-refactor) vs. new (refactored) code, both on CPU -- the algorithmic speedup
     from vectorizing the per-gene loops / removing PyG Data-DataLoader construction / etc.
  2. new code across devices (cpu, mps, cuda) -- the additional speedup from Apple Silicon's
     MPS backend (GPU / unified memory) or CUDA, on top of the refactor.

Each is measured two ways:
  - forward-pass-only (inference: model.eval(), torch.no_grad())
  - full train step (model.train(), forward + a synthetic loss + backward() + optimizer.step()),
    which is what actually matters for training throughput -- the refactor's per-gene-loop
    vectorization affects backward-graph construction/execution too, not just the forward math,
    and device speedup can differ once autograd + optimizer overhead is included.

The synthetic loss (sum of squared-mean over each output head) exists only to produce a
representative backward pass through every parameter -- it is not the real per-task
BCE/MSE multi-task loss `pretrain.py` uses (that needs real labels), and its value is
meaningless; only the timing matters here.

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
    python benchmark_device.py --skip-train-step  # forward-pass-only, faster to run
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
from device_utils import to_device_safe

FI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FI_DIR / 'tests'))  # OldMutationProjector is the frozen pre-refactor baseline

OPTIMIZER_LR = 0.001
OPTIMIZER_WEIGHT_DECAY = 0.0001  # matches src/load_model.py's production hyperparameters


def _sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def _synthetic_loss(output1):
    # exists only to produce a representative backward pass through every parameter; the
    # value itself is meaningless (not the real per-task loss pretrain.py computes)
    return sum(o.float().pow(2).mean() for o in output1)


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


def time_train_step(model, X_converted, X_special, device, n_repeats=5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER_LR, weight_decay=OPTIMIZER_WEIGHT_DECAY)

    def step():
        optimizer.zero_grad()
        output1, masked_positions, gene_emb = model(
            X_converted, X_special_tokens=X_special, test_geneset=False,
            return_attention_weights=False, apply_paddings=False)
        loss = _synthetic_loss(output1)
        loss.backward()
        optimizer.step()

    step()  # warmup
    _sync(device)
    t0 = time.time()
    for _ in range(n_repeats):
        step()
    _sync(device)
    return (time.time() - t0) / n_repeats


def _production_hparams(network_edges, device):
    import numpy as np
    return dict(
        num_genes=468, num_features=10, network_edges=network_edges, num_GATblock=2, num_heads=1,
        dropout_p=0.1, cuda_device=device, output_sizes=[3, 10, 3], mask_percentage=0, input_genes=[],
        d_ff=10, use_representative_embedding=1, ssl_task_index=0, use_special_token=1,
        num_special_tokens=9, num_bins=list(np.append([5, 5], [2] * 7)), use_pooling=0,
    )


def benchmark_old_baseline(network_edges, checkpoint, X_converted, X_special, n_repeats, include_train_step):
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
        results = {}

        model = OldMutationProjector(**hp)
        model.load_state_dict(checkpoint, strict=True)  # old checkpoint format IS the native old-model format
        t = time_forward(model, X_converted, X_special, torch.device('cpu'), n_repeats=n_repeats)
        print(f"  forward-only:  {t * 1000:8.1f} ms/call (avg of {n_repeats} calls)")
        results['forward'] = t

        if include_train_step:
            model = OldMutationProjector(**hp)  # fresh instance: train() + optimizer steps mutate weights
            model.load_state_dict(checkpoint, strict=True)
            t = time_train_step(model, X_converted, X_special, torch.device('cpu'), n_repeats=n_repeats)
            print(f"  train step:    {t * 1000:8.1f} ms/call (avg of {n_repeats} calls)")
            results['train_step'] = t

        return results
    finally:
        torch.Tensor.cuda = orig_tensor_cuda
        torch.nn.Module.cuda = orig_module_cuda


def benchmark_new(network_edges, checkpoint, X_converted, X_special, device_str, n_repeats, include_train_step):
    device = torch.device(device_str)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        print(f"\nskipping new/{device_str}: MPS not available (needs Apple Silicon + a PyTorch build with MPS support)")
        return None
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f"\nskipping new/{device_str}: CUDA not available")
        return None

    print(f"\n=== new (refactored) code, device: {device_str} ===")
    hp = _production_hparams(network_edges, device=device_str)
    X_converted_dev = X_converted.to(device)
    X_special_dev = to_device_safe(X_special, device)
    results = {}

    def build_model():
        model = NewMutationProjector(**hp)
        model.load_state_dict(adapt_legacy_state_dict(model, checkpoint), strict=True)
        return model

    model = build_model()
    sample_param = next(model.parameters())
    print(f"  confirmed model parameters live on: {sample_param.device}")
    t = time_forward(model, X_converted_dev, X_special_dev, device, n_repeats=n_repeats)
    print(f"  forward-only:  {t * 1000:8.1f} ms/call (avg of {n_repeats} calls)")
    results['forward'] = t

    if include_train_step:
        model = build_model()  # fresh instance: train() + optimizer steps mutate weights
        t = time_train_step(model, X_converted_dev, X_special_dev, device, n_repeats=n_repeats)
        print(f"  train step:    {t * 1000:8.1f} ms/call (avg of {n_repeats} calls)")
        results['train_step'] = t

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--devices', nargs='+', default=['cpu', 'mps'],
                         help="new-code devices to benchmark, e.g. cpu mps cuda:0")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--skip-old-baseline', action='store_true',
                         help="skip the pre-refactor code comparison, only benchmark new code across --devices")
    parser.add_argument('--skip-train-step', action='store_true',
                         help="only benchmark forward-pass inference, skip forward+backward+optimizer.step()")
    args = parser.parse_args()
    include_train_step = not args.skip_train_step

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
        results['old/cpu'] = benchmark_old_baseline(network_edges, checkpoint, X_converted, X_special, args.repeats, include_train_step)

    for device_str in args.devices:
        r = benchmark_new(network_edges, checkpoint, X_converted, X_special, device_str, args.repeats, include_train_step)
        if r is not None:
            results[f'new/{device_str}'] = r

    print(f"\n=== summary (batch_size={args.batch_size}) ===")
    header = f"  {'':12s} {'forward':>12s}" + (f"  {'train_step':>12s}" if include_train_step else "")
    print(header)
    for name, r in results.items():
        line = f"  {name:12s} {r['forward']*1000:9.1f} ms"
        if include_train_step and 'train_step' in r:
            line += f"  {r['train_step']*1000:9.1f} ms"
        print(line)

    for mode in (['forward', 'train_step'] if include_train_step else ['forward']):
        if 'old/cpu' in results and 'new/cpu' in results and mode in results['old/cpu'] and mode in results['new/cpu']:
            print(f"\nrefactor speedup, {mode} (new/cpu vs old/cpu): "
                  f"{results['old/cpu'][mode] / results['new/cpu'][mode]:.2f}x")
        if 'new/cpu' in results and mode in results['new/cpu']:
            for name, r in results.items():
                if name not in ('old/cpu', 'new/cpu') and mode in r:
                    print(f"device speedup, {mode} ({name} vs new/cpu): {results['new/cpu'][mode] / r[mode]:.2f}x")


if __name__ == '__main__':
    main()
