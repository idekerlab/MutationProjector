# Training-code performance refactor

## Context

`MutationProjector`'s GATv2 graph encoder (`src/GATv2_functions.py`) and tokenizer
(`src/nn_training_functions.py`) were slow because of per-gene Python loops calling up to 468 individual
`nn.Linear` modules, PyG `Data`/`DataLoader` objects rebuilt from scratch on every forward call
despite static graph topology, a pandas `.loc` lookup and a `pd.cut` round-trip inside
hot loops, and pure-Python per-element loops in mutation tokenization. During the refactor we found
one unrelated correctness bug: a padding path re-creates an untrained `nn.Embedding` on every call.

This refactor changes performance only: identical architecture, identical per-gene
distinct-weight semantics, identical output values (within float tolerance), verified against
the shipped `pretrained_model/pretrained_model.pth`.

## Changes

### 1. `src/nn_training_functions.py`: vectorized `tokenizer.return_mut_embedding`
Replaced a double loop (`for value, key in enumerate(alt_combo): ... for i, j in zip(i_, j_):
out[i][j] = mut_emb`) with a single batched embedding lookup: `self.mut_embedding(X_converted)`
→ `(B, N, F)`, then a masked indexed assignment for masked positions. No parameter changes.

### 2. `src/nn_training_functions.py`: vectorized `convert_mutations`
Replaced an O(B·N) per-element dict-lookup loop with base-2 bit-packing:
`index = (X.long() * (2 ** arange(k-1, -1, -1))).sum(-1)`. This matches
`itertools.product([0,1], repeat=k)`'s enumeration order exactly, because that order *is*
binary counting order, MSB-first. Pure input preprocessing, no model touched.

### 3. `src/GATv2_functions.py`: eliminated the `GATidx` pandas lookup
`GATv2block` used to build a `pd.DataFrame` in `__init__` and query it via a double `.loc`
boolean-mask filter inside the hot `(i, j)` loop, just to map `(i, j)` → a sequential index into
a flat `nn.Sequential` of GAT layers. Since that index is always `i * num_networks + j` by
construction, `GAT_layers` was restructured as a directly `(i, j)`-indexable 2D `nn.ModuleList`
(`self.GAT_layers[i][j]`), so no lookup structure is needed.

### 4. `src/GATv2_functions.py`: eliminated per-forward `Data`/`DataLoader` construction
This change gives the largest speedup. `GATv2block.forward` used to build a fresh list of PyG `Data` objects and a
fresh `DataLoader` on every `(GAT block, network)` iteration (up to 16 times per forward call in
the production 2-block/8-network config), even though the graph topology is static and
`batch_size == num_samples` always (so the loader only ever yielded one batch).

Replaced with a block-diagonal offset `edge_index`, cached per `(network, batch_size)`:
`cat([edge_index + k*num_nodes for k in range(batch_size)])` (what PyG's
`Batch.from_data_list` computes internally), plus a plain `.reshape()` of the node features.
Zero PyG `Data`/`DataLoader` objects touched per forward call.

### 5. `src/nn_training_functions.py`: replaced `pd.cut` with `torch.bucketize`
Removed a GPU→CPU→pandas→GPU round-trip in `tokenize_special_tokens.forward`. `pd.cut(bins=int)`
computes **adaptive, batch-dependent** bin edges (min/max of the current input, widened by 0.1%
of the range, right-closed intervals). The new code replicates this exactly with `np.linspace`
(see the note below) plus `torch.bucketize`, sending only the tiny `num_bins+1`
edge array through numpy instead of the whole batch.

> During verification, `torch.linspace` and `np.linspace` were found to
> disagree by a few float64 ULPs on the same `(start, stop, n)`. That's invisible for almost all
> inputs, but if a covariate value lands *exactly* on a computed bin edge, it can push the value
> into the adjacent bin in one implementation but not the other. Using `np.linspace` for the
> edge computation (bit-identical to what `pd.cut` uses internally) eliminates this. 500+ adversarial trials with values placed exactly on bin boundaries confirm this.

### 6. Padding `nn.Embedding` bug (left unfixed)
`tokenize_special_tokens.forward`'s `apply_padding==True` branch instantiates a fresh
`nn.Embedding(1, num_features, padding_idx=0)` on every call instead of a persistent `__init__`
submodule. We left this unchanged, because fixing it changes numeric output for the
3 datasets that hit this path (`IMvigor210`, `mel_dfci_2019`, `mixed_allen_2018`). That is out of scope
for a no-semantic-change refactor. It's documented in code and pinned by a characterization test
(see below) so a future change can't silently alter this behavior.

During testing we found that PyTorch always zeroes the single embedding row (`num_embeddings=1`,
`padding_idx=0`) for any random seed. The padding embedding therefore stays at zero and receives
no training.

### 7. Removed confirmed-redundant `.cuda()` calls
Commit 7388522 removed a net 10 `.cuda()` calls (9 in `MutationProjector_nn.py`, 1 in
`nn_training_functions.py`), some of them as part of items 1 and 2. The removed calls moved tensors
that were already on `cuda_device`, for example the output of a submodule whose weights are on
`cuda_device`. We traced each tensor's origin and left the ambiguous sites unchanged.

### 8. Per-gene `Linear` → `BatchedPerGeneLinear`
The highest-risk change. `GATv2block.linear_layers`/`FF_layer1`/`FF_layer2` and
`MutationProjector`'s SSL head `prot2gene_layer` were each a `nn.ModuleList` of up to 468
independent `nn.Linear` layers, applied via a Python loop (one tiny GEMM per gene). We also removed
an `X.clone()` that copied the full tensor in each of the 468 iterations, although the loop read
only a per-gene slice.

Replaced with `BatchedPerGeneLinear` (`src/GATv2_functions.py`): a single `nn.Parameter` of shape
`(num_genes, out_features, in_features)` (+ bias), computed via one
`torch.einsum('bgi,goi->bgo', x, weight) + bias` instead of the loop.

The checkpoint adapter handles the legacy keys. The shipped checkpoint stores individual per-gene keys
(`GATblock.linear_layers.0.weight ... .467.weight`, etc.) and a flat `GAT_layers` indexed
`i*num_networks+j`. These differ from the new shapes. `adapt_legacy_state_dict` (`src/load_model.py`) detects
these legacy key patterns via regex, stacks per-gene weights into the new tensor shape, and
re-nests `GAT_layers` keys. It passes everything else through unchanged. The adapter translates the keys at
load time and leaves the checkpoint file unchanged. `src/load_model.py` and
`src/generate_embeddings.py`'s load sites now route through this adapter instead of calling
`load_state_dict` directly.

## Verification

We compared each changed component and the full model with frozen pre-refactor reference implementations
(`tests/reference_impls.py`) with randomized equivalence tests, plus a real end-to-end run
against the actual shipped checkpoint:

- Loaded the real `pretrained_model.pth` (468 genes, 8 networks, 9 special tokens, 3927 state
  dict keys) into both the reference architecture and the refactored architecture (via the
  adapter). All keys matched, with zero missing or unexpected keys in both.
- Ran both models on the real `data/downstream_data/eval_dataset/test_sample` data.
  `output1`, `gene_emb`, `rep_emb`, `cov_emb`, `out_concat_layer`, and all 16 GAT layers'
  attention weights/edge indices matched exactly (max diff 0.0).
- Cross-checked against the checked-in `prediction_results/test_sample/*.pt` reference
  embeddings (matches within 5e-6 for both old and new).
- Timed CPU runs: 1.74x to 2.49x speedup.
- Timed runs on one NVIDIA A30 GPU, with old and new code both on `cuda:0`: 52.4x (batch 64) and
  52.0x (batch 256) for the forward pass, and 44.4x and 48.5x for a full train step.

## Test suite (`tests/`)

Run with `pytest tests/` from the repo root (`pytest.ini` wires `src/` and `tests/` onto
`sys.path`). The slow test is marked `@pytest.mark.slow` and uses the checkpoint, network and
test_sample files tracked in the repo.

| File | Covers |
|---|---|
| `conftest.py` | `.cuda()` no-op for the old code, seeded RNG, tiny model config and builders |
| `reference_impls.py` | Frozen pre-refactor implementations, the equivalence baseline |
| `test_convert_mutations.py` | Item 2 |
| `test_return_mut_embedding.py` | Item 1, incl. masking edge cases |
| `test_batched_edge_index.py` | Items 3+4, incl. the `compute_attention.py` downstream consumer |
| `test_special_token_binning.py` | Item 5, incl. 200 exact-boundary trials |
| `test_padding_bug_documented.py` | Item 6 characterization test |
| `test_batched_per_gene_linear.py` | Item 8 core numerics |
| `test_state_dict_adapter.py` | Item 8 checkpoint key translation, incl. passthrough of new-format keys |
| `test_full_model_equivalence.py` | Full model, 4 config branches (covers item 7 too) |
| `test_device_utils.py` | Legacy int device index, MPS float64 downcast (MPS test runs only on Apple Silicon) |
| `test_end_to_end_checkpoint.py` *(slow)* | Real 468-gene checkpoint, real data |
| `test_mp2_cohort.py` | MP2 cohort conversion, fixed bin edges, per-sample covariate padding. The slow test checks that Run B reproduces Run A on test_sample |

Nothing outside `src/`, `tests/`, and `pytest.ini` was changed; `pretrained_model.pth` is
unchanged.

## Device support: Apple Silicon (MPS) / CPU / CUDA

The codebase originally hardcoded `.cuda(device)` everywhere, which only works on NVIDIA GPUs
and raises immediately on any machine without CUDA (including Apple Silicon). This is now
device-agnostic:

- `src/device_utils.py` adds `resolve_device(device)`, which normalizes a plain int (the
  historical `.cuda(0)` convention) to a CUDA device index for backward compatibility, or passes
  through any native device spec: `'mps'` (Apple Silicon GPU / unified memory), `'cpu'`,
  `'cuda:0'`, or a `torch.device`.
- Every `.cuda(device)` call site in `GATv2_functions.py`, `MutationProjector_nn.py`,
  `nn_training_functions.py`, and `generate_embeddings.py` was replaced with `.to(device)`
  (where `device` is resolved via `resolve_device` once, at construction time).
- `torch.load(...)` calls for the checkpoint, network edge files, and saved embeddings
  (`load_model.py`, `generate_embeddings.py`, `import_network.py`, `transfer_learn.py`,
  `use_transfer_learned_model.py`) now pass `map_location='cpu'`. Without it, loading a
  checkpoint saved from CUDA tensors raises on any non-CUDA machine, before device placement
  runs.
- `load_model.py`'s `load_MutationProjector(device=0)` now accepts an optional `device` override
  (still defaults to legacy CUDA index `0` for backward compatibility).
- `src/benchmark_device.py` measures both forward-pass-only inference and a full train step
  (forward + synthetic loss + backward + optimizer.step()) for old (pre-refactor, CPU-only via
  a `.cuda()` shim) vs. new (refactored) code, and for new code across devices (cpu/mps/cuda),
  using the real checkpoint and shipped test data. See below for Mac usage.

### Testing the speedup on Apple Silicon (M-series GPU / unified memory)

1. **Environment** (native `uv` venv: `conda-envs/env.yml` pins CUDA, so it fails on macOS):
   ```
   uv venv .venv && source .venv/bin/activate
   uv pip install torch  # macOS wheels include MPS support out of the box
   uv pip install torch_geometric pandas==1.5.3 scikit-learn==1.3.2 scipy==1.13.1 numpy matplotlib seaborn joblib ndex2 networkx
   ```
2. **Run the benchmark** from inside `src/` (this repo's scripts resolve paths relative to CWD;
   the venv from step 1 should already be activated):
   ```
   cd src
   python benchmark_device.py --devices cpu mps --batch-size 64 --repeats 10
   ```
   This loads the real `pretrained_model.pth`, times both a forward-only pass and a full train
   step on each device (each with a warmup call first), and prints speedup ratios for both. It
   also prints the device of the model's parameters to confirm the run stayed on MPS. Pass `--skip-train-step` for a faster,
   forward-only run.
3. **Confirm GPU utilization independently**: open Activity Monitor → Window → GPU History
   while the benchmark runs, or watch `sudo powermetrics --samplers gpu_power` in another
   terminal. You should see GPU activity spike during the `mps` run and stay flat during `cpu`.
4. If you hit "operator not implemented for MPS" errors: some PyTorch/PyG ops still have gaps
   in MPS coverage depending on your PyTorch version. Set
   `PYTORCH_ENABLE_MPS_FALLBACK=1` as an environment variable to let those specific ops
   fall back to CPU automatically. The rest still runs on MPS.
