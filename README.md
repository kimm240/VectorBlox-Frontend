# VectorBlox Frontend

[![ONNX numerical correctness](https://github.com/kimm240/VectorBlox-Frontend/actions/workflows/onnx-numerical.yml/badge.svg)](https://github.com/kimm240/VectorBlox-Frontend/actions/workflows/onnx-numerical.yml)

A frontend for VectorBlox using MLIR.

- Rewrites operators currently unsupported by the VectorBlox SDK into combinations of supported operators.

## Build

Use Linux with a C++17 compiler, CMake 3.20+, Ninja and Python 3. For example:

```sh
sudo apt-get install build-essential cmake ninja-build python3 python3-venv
python3 tools/bootstrap.py --jobs 2
build/vbx-rewrite --help
```

For an offline build, first populate `.deps/downloads`, then run:

```sh
python3 tools/bootstrap.py --offline --jobs 2
```

See `third_party/dependencies.lock.json`, [third-party notices](docs/THIRD_PARTY_NOTICES.md), and
[dependency patches](third_party/patches/README.txt) for versions and provenance.

## Legalization

VectorBlox SDK 3.1 cannot directly compile certain ONNX patterns, such as channel-wise mean reduction and static unfold.
MLIR legalization passes rewrite these patterns into combinations of SDK-supported operators.

### 1. Verified models

Conversion and execution have been verified for the following configurations:

| Model | Verified configuration |
| --- | --- |
| [XFeat](https://github.com/verlab/accelerated_features) | Dense forward, FP32, static `1×3×256×256`, ONNX opset 17 |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Small / ViT-S, FP32, static `1×3×224×224`, ONNX opset 17 |

### 2. Run ONNX legalization

```sh
build/vbx-rewrite input.onnx legalized.onnx \
  --report rewrite-report.json \
  --dump-mlir legalized.mlir
```

See [legalization passes](src/Frontend/ONNX/README.md) for supported patterns and their conditions.

The tool writes a new model; input and output must be different files. For ONNX
external tensor data, keep the rewritten model in the original model directory:
external weight files are not copied by this tool. `--max-channels` and
`--max-rewrites` bound the applicable channel policy and rewrite count.

### 3. Check numerical correctness

```sh
python -m pip install -r tests/requirements.txt
python tests/legalization/compare_onnx.py input.onnx legalized.onnx --report numerical-report.json
```

Compares both models' outputs using the same inputs. Prints `PASS` when outputs match within tolerance, or `FAIL` otherwise. Detailed results are saved to `numerical-report.json`.

## Contributing

**Unsupported models and ideas for useful MLIR frontend work are welcome.**
We are particularly interested in reproducible failures, robust legalization,
layout simplification and opportunities to reuse existing MLIR passes or
interfaces. A proposed optimization should explain its safety conditions and
how numerical equivalence and the intended benefit can be measured.

Use the [unsupported-model form](https://github.com/kimm240/VectorBlox-Frontend/issues/new?template=unsupported-model.yml)
or the [MLIR frontend idea form](https://github.com/kimm240/VectorBlox-Frontend/issues/new?template=frontend-idea.yml).
A minimal model report can be as simple as:

```text
Model / variant / upstream URL and version:
Input names, shapes and dtypes:
ONNX opset and exporter version:
Frontend commit / SDK version:
Command and failing stage:
Error log or rewrite report:
Minimal ONNX reproducer or export script:
Expected result / before-after numerical report, if available:
```

For a code contribution, add a small before/after numerical regression and
rejection cases for unsupported shapes or semantics. Keep model-specific names
out of rewrite matching. Open a pull request with the reproducer, the rule's
preconditions and test results.

## License

Original project code is licensed under [Apache-2.0](LICENSE). Third-party
components retain their own licenses; see [third-party notices](docs/THIRD_PARTY_NOTICES.md).
