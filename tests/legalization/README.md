# Legalization tests

Run commands from the repository root after building the frontend and installing
`tests/requirements.txt`. These tests are separate from normal conversion.

## Pattern regressions

```sh
python tests/legalization/validate_numerical_comparator.py
python tests/legalization/validate_onnx.py
python tests/legalization/validate_unfold.py
python tests/legalization/validate_legalizations.py
python tests/legalization/validate_model_runner.py
```

| Suite | Coverage |
| --- | --- |
| `rewrite_test.cpp`, `validate_onnx.py` | Channel mean, legality guards, external data and numerical equivalence |
| `validate_unfold.py` | Static unfold and rejection guards |
| `validate_legalizations.py` | Linear/attention, LayerNorm, GELU, Softmax, Gather, Slice and ConvTranspose patterns |
| `validate_numerical_comparator.py` | Reject incorrect values, output names/shapes and nonfinite outputs |
| `validate_model_runner.py` | Synthetic controls for the full-model test runner; not evidence that pretrained models pass |

C++ checks run through `ctest --test-dir build --output-on-failure` after
`python3 tools/bootstrap.py --test`. GitHub Actions runs these generated fixtures;
weights and full models are not downloaded by CI.

## Models

### Generate ONNX and validate end to end

```sh
python3 -m venv .venv-models
. .venv-models/bin/activate
python -m pip install -r tests/legalization/requirements-models.txt
python tests/legalization/prepare_models.py
```

This downloads official sources and weights pinned in `model_sources.json`,
checks their revisions/hashes, and reproduces the previous PyTorch export route:

```text
Official PyTorch model + pretrained weights
  -> torch.onnx.export (opset 17)
  -> ONNX shape inference (intermediate shapes/types)
  -> original ONNX
  -> C++ MLIR legalization
  -> rewritten ONNX
  -> compare original / rewritten outputs
```

This follows the SDK tutorial's PyTorch-to-ONNX export approach; the SDK does not
provide dedicated XFeat or Depth Anything V2 tutorials. No SDK installation,
TFLite conversion or VNNX generation is involved. PyTorch-to-ONNX output
agreement is checked before legalization. A failure at either stage fails the run.

The default output/cache directory is `artifacts/model-validation/`. It retains
raw exports (`exported.onnx`), shape-annotated models (`model.onnx`), export logs, source/weight provenance, export-check
results and the final `results.json`; rewrite and numerical reports live in
`comparison/`. Original ONNX files and downloaded weights are ignored by Git.
Use `--out` to choose another directory. Re-running reuses verified downloads
and re-exports both models. Dependencies are isolated in `.venv-models`.

For GitHub Actions, choose **Run workflow** and enable **full_models**. Normal
push/PR CI runs the generated fixtures without model downloads.

### Validate existing ONNX exports

`validate_models.py` converts both supplied ONNX models, requires at least one
rewrite per model, and checks original versus rewritten outputs using identical
inputs. Both must pass; a missing model, failed conversion, zero rewrites or
numerical mismatch returns a nonzero exit code.

```sh
python tests/legalization/validate_models.py \
  --xfeat /path/to/xfeat.onnx \
  --depth-anything-v2 /path/to/depth-anything-v2.onnx
```

Use XFeat dense FP32 `1×3×256×256` and Depth Anything V2 Small/ViT-S FP32
`1×3×224×224`, exported with ONNX opset 17. Supply your own exports and weights;
the runner does not verify the model's identity from its filename.
It replaces the old optional `validate_onnx.py --xfeat` check.

By default each model gets three deterministic uniform `[0,1)` inputs, with
`atol=1e-4` and `rtol=1e-3` (configurable). For actual preprocessed model inputs,
use `--xfeat-inputs sample.npz` and `--depth-inputs sample.npz`; NPZ keys must
match ONNX input names. This checks sampled numerical equivalence, not task accuracy.
ONNX Runtime graph optimizations are disabled and all outputs must match.

Reports are saved under `artifacts/legalization-models/`: `results.json`,
`<model>.rewrite.json` and `<model>.numerical.json` when those stages run.
Temporary rewritten models are created beside the supplied models to preserve
external-weight references, then removed. The model directories must be writable;
original models and weights are not modified.

## Compare any existing before/after pair

```sh
python tests/legalization/compare_onnx.py input.onnx legalized.onnx --report numerical-report.json
```

Pattern tests use `atol=rtol=1e-4`, channel mean uses `3e-5`, and unfold requires
exact equality. The comparator rejects nonfinite outputs and shape/dtype mismatches.

