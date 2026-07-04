# Training-code performance refactor

## Context

`MutationProjector`'s GATv2 graph encoder (`src/GATv2_functions.py`) and tokenizer
(`src/nn_training_functions.py`) were slow mostly because of *implementation* choices, not the
graph-attention architecture itself: per-gene Python loops calling hundreds of individual
`nn.Linear` modules, PyG `Data`/`DataLoader` objects rebuilt from scratch on every forward call
despite static graph topology, a pandas `.loc` lookup and a `pd.cut` round-trip sitting inside
hot loops, and pure-Python per-element loops in mutation tokenization. One unrelated correctness
bug (an untrained `nn.Embedding` re-created every call in a padding path) was found along the way.

This refactor is a **pure performance change**: identical architecture, identical per-gene
distinct-weight semantics, identical output values (within float tolerance), verified against
the shipped `pretrained_model/pretrained_model.pth`.

## Changes

### 1. `src/nn_training_functions.py` — vectorized `tokenizer.return_mut_embedding`
Replaced a double loop (`for value, key in enumerate(alt_combo): ... for i, j in zip(i_, j_):
out[i][j] = mut_emb`) with a single batched embedding lookup: `self.mut_embedding(X_converted)`
→ `(B, N, F)`, then a masked indexed assignment for masked positions. No parameter changes.

### 2. `src/nn_training_functions.py` — vectorized `convert_mutations`
Replaced an O(B·N) per-element dict-lookup loop with base-2 bit-packing:
`index = (X.long() * (2 ** arange(k-1, -1, -1))).sum(-1)`. This matches
`itertools.product([0,1], repeat=k)`'s enumeration order exactly, because that order *is*
binary counting order, MSB-first. Pure input preprocessing, no model touched.

### 3. `src/GATv2_functions.py` — eliminated the `GATidx` pandas lookup
`GATv2block` used to build a `pd.DataFrame` in `__init__` and query it via a double `.loc`
boolean-mask filter inside the hot `(i, j)` loop, just to map `(i, j)` → a sequential index into
a flat `nn.Sequential` of GAT layers. Since that index is always `i * num_networks + j` by
construction, `GAT_layers` was restructured as a directly `(i, j)`-indexable 2D `nn.ModuleList`
(`self.GAT_layers[i][j]`) — no lookup structure needed at all.

### 4. `src/GATv2_functions.py` — eliminated per-forward `Data`/`DataLoader` construction
The biggest win. `GATv2block.forward` used to build a fresh list of PyG `Data` objects and a
fresh `DataLoader` on every `(GAT block, network)` iteration (up to 16 times per forward call in
the production 2-block/8-network config), even though the graph topology is static and
`batch_size == num_samples` always (so the loader only ever yielded one batch).

Replaced with a block-diagonal offset `edge_index`, cached per `(network, batch_size)`:
`cat([edge_index + k*num_nodes for k in range(batch_size)])` — exactly what PyG's
`Batch.from_data_list` computes internally — plus a plain `.reshape()` of the node features.
Zero PyG `Data`/`DataLoader` objects touched per forward call.

### 5. `src/nn_training_functions.py` — replaced `pd.cut` with `torch.bucketize`
Removed a GPU→CPU→pandas→GPU round-trip in `tokenize_special_tokens.forward`. `pd.cut(bins=int)`
computes **adaptive, batch-dependent** bin edges (min/max of the current input, widened by 0.1%
of the range, right-closed intervals) — this was replicated exactly using `np.linspace` (not
`torch.linspace` — see note below) plus `torch.bucketize`, sending only the tiny `num_bins+1`
edge array through numpy instead of the whole batch.

> **Numerical subtlety found during verification:** `torch.linspace` and `np.linspace` can
> disagree by a few float64 ULPs on the same `(start, stop, n)`. That's invisible for almost all
> inputs, but if a covariate value lands *exactly* on a computed bin edge, it can push the value
> into the adjacent bin in one implementation but not the other. Using `np.linspace` for the
> edge computation (bit-identical to what `pd.cut` uses internally) eliminates this — confirmed
> with 500+ adversarial trials where sampled values were placed exactly on bin boundaries.

### 6. Padding `nn.Embedding` bug — flagged, not fixed
`tokenize_special_tokens.forward`'s `apply_padding==True` branch instantiates a fresh
`nn.Embedding(1, num_features, padding_idx=0)` on every call instead of a persistent `__init__`
submodule. Per decision, this is **left as-is** (fixing it would change numeric output for the
3 datasets that hit this path: `IMvigor210`, `mel_dfci_2019`, `mixed_allen_2018`) — out of scope
for a no-semantic-change refactor. It's documented in code and pinned by a characterization test
(see below) so a future change can't silently alter this behavior.

Note found during testing: because `num_embeddings=1` and `padding_idx=0`, PyTorch always zeroes
that single embedding row regardless of random seed — so in practice this bug doesn't produce
random output, it produces a permanently-stuck-at-zero, never-trainable padding embedding.

### 7. Removed confirmed-redundant `.cuda()` calls
A handful of call sites in `MutationProjector_nn.py` and `nn_training_functions.py` called
`.cuda(self.cuda_device)` on tensors that were already guaranteed to be on that device (e.g. the
output of a submodule whose weights are already on `cuda_device`). Only sites confirmed redundant
by tracing the tensor's origin were touched; ambiguous ones were left alone.

### 8. Per-gene `Linear` → `BatchedPerGeneLinear`
The highest-risk change. `GATv2block.linear_layers`/`FF_layer1`/`FF_layer2` and
`MutationProjector`'s SSL head `prot2gene_layer` were each a `nn.ModuleList` of up to 468
independent `nn.Linear` layers, applied via a Python loop (one tiny GEMM per gene). Also removed
an `X.clone()` that was redundantly re-cloning the full tensor on every one of those 468
iterations even though only a read-only per-gene slice was used.

Replaced with `BatchedPerGeneLinear` (`src/GATv2_functions.py`): a single `nn.Parameter` of shape
`(num_genes, out_features, in_features)` (+ bias), computed via one
`torch.einsum('bgi,goi->bgo', x, weight) + bias` instead of the loop.

**Checkpoint bridge:** the shipped checkpoint stores individual per-gene keys
(`GATblock.linear_layers.0.weight ... .467.weight`, etc.) and a flat `GAT_layers` indexed
`i*num_networks+j`, not the new shapes. `adapt_legacy_state_dict` (`src/load_model.py`) detects
these legacy key patterns via regex, stacks per-gene weights into the new tensor shape, and
re-nests `GAT_layers` keys — passing everything else through unchanged. **The checkpoint file
itself is never modified**, only translated at load time. `src/load_model.py` and
`src/generate_embeddings.py`'s load sites now route through this adapter instead of calling
`load_state_dict` directly.

## Verification

Everything was checked against frozen pre-refactor reference implementations
(`tests/reference_impls.py`) with randomized equivalence tests, plus a real end-to-end run
against the actual shipped checkpoint:

- Loaded the real `pretrained_model.pth` (468 genes, 8 networks, 9 special tokens, 3927 state
  dict keys) into both the reference architecture and the refactored architecture (via the
  adapter) — all keys matched with zero missing/unexpected in both.
- Ran both models on the real `data/downstream_data/eval_dataset/test_sample` data —
  `output1`, `gene_emb`, `rep_emb`, `cov_emb`, `out_concat_layer`, and all 16 GAT layers'
  attention weights/edge indices matched **exactly (max diff 0.0)**.
- Cross-checked against the checked-in `prediction_results/test_sample/*.pt` reference
  embeddings (matches within 5e-6 for both old and new).
- Timing: **1.74x–2.49x speedup** even on a CPU-only sandbox with no GPU kernel-launch overhead
  to amortize (the actual production speedup on GPU, with 468 genes' worth of eliminated kernel
  launches, should be substantially larger).

## Test suite (`tests/`)

Run with `pytest tests/` from the repo root (`pytest.ini` wires `src/` and `tests/` onto
`sys.path`). Fast tests (~300, a few seconds) run by default; full-scale tests are marked
`@pytest.mark.slow` and require the real checkpoint/network/data files that ship with the repo:

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures: CPU-only `.cuda()` monkeypatch (no GPU in CI), seeded RNG, tiny synthetic model config |
| `reference_impls.py` | Frozen pre-refactor implementations used as the equivalence baseline everywhere else |
| `test_convert_mutations.py` | Item 2, incl. the bit-order assumption |
| `test_return_mut_embedding.py` | Item 1, incl. masking edge cases |
| `test_gatidx_indexing.py` | Item 3's arithmetic-index invariant |
| `test_batched_edge_index.py` | Items 3+4, incl. the `compute_attention.py` downstream consumer |
| `test_special_token_binning.py` | Item 5, incl. 200+ exact-boundary adversarial cases |
| `test_padding_bug_documented.py` | Item 6, characterization/tripwire test |
| `test_batched_per_gene_linear.py` | Item 8 core numerics |
| `test_state_dict_adapter.py` | Item 8 checkpoint bridge, incl. idempotency on already-new-format keys |
| `test_full_model_equivalence.py` | Full model, 4 config branches (covers item 7 too) |
| `test_end_to_end_checkpoint.py` *(slow)* | Real 468-gene checkpoint, real data |
| `test_timing_regression.py` *(slow)* | Synthetic + real-checkpoint speedup check |

Nothing outside `src/`, `tests/`, and `pytest.ini` was changed; `pretrained_model.pth` itself is
untouched.

## Device support: Apple Silicon (MPS) / CPU / CUDA

The codebase originally hardcoded `.cuda(device)` everywhere, which only works on NVIDIA GPUs
and raises immediately on any machine without CUDA (including Apple Silicon). This is now
device-agnostic:

- `src/device_utils.py` adds `resolve_device(device)`, which normalizes a plain int (the
  historical `.cuda(0)` convention) to a CUDA device index for backward compatibility, or passes
  through any native device spec — `'mps'` (Apple Silicon GPU / unified memory), `'cpu'`,
  `'cuda:0'`, or a `torch.device`.
- Every `.cuda(device)` call site in `GATv2_functions.py`, `MutationProjector_nn.py`,
  `nn_training_functions.py`, and `generate_embeddings.py` was replaced with `.to(device)`
  (where `device` is resolved via `resolve_device` once, at construction time).
- `torch.load(...)` calls for the checkpoint, network edge files, and saved embeddings
  (`load_model.py`, `generate_embeddings.py`, `import_network.py`, `transfer_learn.py`,
  `use_transfer_learned_model.py`) now pass `map_location='cpu'` — without this, loading a
  checkpoint saved from CUDA tensors raises on any non-CUDA machine, before device placement
  even gets a chance to run.
- `load_model.py`'s `load_MutationProjector(device=0)` now accepts an optional `device` override
  (still defaults to legacy CUDA index `0` for backward compatibility).
- `src/benchmark_device.py` is a standalone script to measure forward-pass latency across
  devices using the real checkpoint and shipped test data — see below for Mac usage.

### Testing the speedup on Apple Silicon (M-series GPU / unified memory)

1. **Environment** (native pip install, not the CUDA-pinned `conda-envs/env.yml`):
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch  # macOS wheels include MPS support out of the box
   pip install torch_geometric pandas==1.5.3 scikit-learn==1.3.2 scipy==1.13.1 numpy matplotlib seaborn joblib ndex2 networkx
   ```
2. **Run the benchmark** from inside `src/` (this repo's scripts resolve paths relative to CWD):
   ```
   cd src
   python benchmark_device.py --devices cpu mps --batch-size 64 --repeats 10
   ```
   This loads the real `pretrained_model.pth`, times a forward pass on each device (with a
   warmup call first), and prints a speedup ratio. It also prints the actual device of the
   model's parameters as a sanity check that MPS is really being used, not silently falling
   back to CPU.
3. **Confirm GPU utilization independently**: open Activity Monitor → Window → GPU History
   while the benchmark runs, or watch `sudo powermetrics --samplers gpu_power` in another
   terminal — you should see GPU activity spike during the `mps` run and stay flat during `cpu`.
4. If you hit "operator not implemented for MPS" errors: some PyTorch/PyG ops still have gaps
   in MPS coverage depending on your PyTorch version. Set
   `PYTORCH_ENABLE_MPS_FALLBACK=1` as an environment variable to let those specific ops
   fall back to CPU automatically rather than crashing (the rest still runs on MPS).
