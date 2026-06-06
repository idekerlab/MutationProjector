# MutationProjector Docker Prediction Smoke Test

The Docker image entrypoint runs `src/predict.py`, so prediction flags can be
passed directly to the image.

Run this from the repository root on a CUDA/GPU Docker host:

```bash
mkdir -p prediction_results

docker run --rm --gpus all \
  -v "$PWD/prediction_results:/opt/MutationProjector/prediction_results" \
  digitaltumors/mutationprojector:0.1.0 \
  -downstream_eval test_sample \
  -transfer_learned_model Immunotherapy \
  -cuda_device 0 \
  -o smoke_test
```

Verify the output:

```bash
head prediction_results/test_sample/smoke_test.txt
```

The output should be a tab-delimited file with columns like:

```text
sample  pred_proba
```

Notes:

- Example eval data is at `data/downstream_data/eval_dataset/test_sample`.
- Output is written under `prediction_results/test_sample`.
- Valid `-transfer_learned_model` values include `Chemotherapy`,
  `Immunotherapy`, `metastasis_luad`, and the tissue-of-origin models.
- Full prediction requires CUDA because the code calls `.cuda(...)`
  unconditionally. On Apple Silicon or non-GPU Docker Desktop, `--help` can
  verify the image entrypoint, but actual prediction is not a meaningful test.

To verify the image entrypoint and available arguments without running a full
prediction:

```bash
docker run --rm --platform linux/amd64 \
  digitaltumors/mutationprojector:0.1.0 \
  --help
```
