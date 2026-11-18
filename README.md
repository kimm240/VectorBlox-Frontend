# VectorBlox Frontend

[![ONNX numerical correctness](https://github.com/kimm240/VectorBlox-Frontend/actions/workflows/onnx-numerical.yml/badge.svg)](https://github.com/kimm240/VectorBlox-Frontend/actions/workflows/onnx-numerical.yml)

A frontend for VectorBlox using MLIR.

- Rewrites operators currently unsupported by the VectorBlox SDK into combinations of supported operators.

## Supported model cases

The retained rules were developed and exercised with these model configurations:

| Model | Reference configuration | Main legalization patterns |
| --- | --- | --- |
| [XFeat](https://github.com/verlab/accelerated_features) | Dense forward export, FP32, static `1×3×256×256`, ONNX opset 17 | Channel mean and static unfold |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Small / ViT-S relative depth, FP32, static `1×3×224×224`, ONNX opset 17 | Linear/attention, LayerNorm, Softmax, GELU, Gather, layout and ConvTranspose |

These are reference cases, not a guarantee for every exporter version, model
variant, input size or quantization configuration. In particular, the depth
case uses a reduced input size rather than the default 518-pixel setting.
Weights and exported models are not distributed here. CI checks self-contained
ONNX reproductions of the retained rules; it does not download or run these
complete pretrained models or certify task-level accuracy.

## Build

Use Linux with a C++17 compiler, CMake 3.20+, Ninja and Python 3. For example:

```sh
sudo apt-get install build-essential cmake ninja-build python3 python3-venv
python3 tools/bootstrap.py --jobs 2
build/vbx-rewrite --help
```

Bootstrap downloads hash-pinned, selected LLVM/MLIR and Protobuf sources into
`.deps` and builds inside this checkout. It does not require a system LLVM/MLIR
installation or download every LLVM backend. A cold build takes substantially
longer than running a rewrite. Increase `--jobs` if the host has sufficient RAM.

For an offline build, first populate `.deps/downloads`, then run:

```sh
python3 tools/bootstrap.py --offline --jobs 2
```

See `dependencies.lock.json`, [third-party notices](THIRD_PARTY_NOTICES.md), and
[dependency patches](third_party/patches/README.txt) for versions and provenance.

## Run ONNX legalization

```sh
build/vbx-rewrite input.onnx legalized.onnx \
  --report rewrite-report.json \
  --dump-mlir legalized.mlir
```

The rules include channel `ReduceMean → Conv`, static unfold, guarded
MatMul/Gemm and attention lowering, LayerNormalization decomposition, Softmax
and GELU tiling, scalar-Mul lowering, Gather, high-rank Slice and selected
ConvTranspose decomposition. Shape, dtype, opset, cost and metadata guards limit
each rule to supported forms. Nested graphs/functions are not rewritten.

The tool writes a new model; input and output must be different files. For ONNX
external tensor data, keep the rewritten model in the original model directory:
external weight files are not copied by this tool. `--max-channels` and
`--max-rewrites` bound the applicable channel policy and rewrite count.

## Check numerical correctness

Create a separate Python environment for tests:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-test.txt
python tests/compare_onnx.py input.onnx legalized.onnx \
  --samples 10 --edge-cases --atol 1e-4 --rtol 1e-4 \
  --report numerical-report.json
```

The comparator runs both models with identical inputs using ONNX Runtime's CPU
provider with graph optimizations disabled. It matches named inputs/outputs,
checks shapes and dtypes, and applies
`abs(after - before) <= atol + rtol * abs(before)` to floating outputs.
Non-floating outputs require exact equality; nonfinite outputs fail. A mismatch
returns a nonzero exit code.

Generated random and zero/one inputs are appropriate only when the model's input
domain permits them. For dynamic shapes, non-floating inputs, or realistic model
inputs, supply NPZ files mapping input names to NumPy arrays:

```sh
python tests/compare_onnx.py input.onnx legalized.onnx \
  --inputs sample1.npz sample2.npz --report numerical-report.json
```

[GitHub Actions](.github/workflows/onnx-numerical.yml) builds the native tool,
creates ONNX fixtures, applies legalization and compares before/after outputs.
It also checks that intentionally incorrect outputs are rejected and runs SDK
layout guard tests. Numerical reports are uploaded as workflow artifacts even
when a later test fails. Channel-mean tests use `atol=rtol=3e-5`, unfold requires
exact equality, and general legalization tests use `atol=rtol=1e-4` with random
and constant inputs. This is sampled equivalence testing, not a proof for every
input or a measurement of INT8/task accuracy.

To run the same standalone suites locally, see [tests/README.md](tests/README.md).
Tests never run inside a normal compiler invocation. CI does not compile VNNX
or connect to a board.

## Optional VectorBlox SDK integration

ONNX-to-TFLite export/quantization is a separate step using the SDK's supported
tooling. Given a prepared TFLite model, the optional adapter targets official
VectorBlox SDK `release-v3.1`, commit
`f5a428c3c0ee7e06ad06657c5fcdcc1eda8cf046`.

```sh
git clone --branch release-v3.1 --depth 1 \
  https://github.com/Microchip-Vectorblox/VectorBlox-SDK.git .deps/vectorblox-sdk-3.1
# Set up and activate that SDK's Python environment using its instructions.
python tools/compile_sdk.py model.tflite -o model.vnnx
```

The adapter checks the SDK commit and simulator hash against
`targets/vectorblox-3.1.json`, applies C++ layout legalization and invokes SDK
VNNX generation. It leaves SDK sources untouched. To run just the C++ stage on
an SDK flatc JSON graph:

```sh
build/vbx-rewrite --sdk-json before.json after.json --report layout-report.json
```

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

## Project layout

- `src/Frontend/ONNX/`: ONNX import/export, legalization pass and patterns.
- `src/Target/VectorBlox/`: SDK JSON/SSA bridge and layout legalization.
- `src/Tools/`: `vbx-rewrite` CLI.
- `include/vbx_rewrite/`: public C++ API.
- `tools/`: pinned dependency bootstrap and optional SDK adapter.
- `tests/`: standalone numerical and guard tests.

See [architecture](docs/architecture.md) for implementation boundaries. This
repository does not include the discontinued memory-scheduling experiments,
model archives, tutorial sweeps or board benchmark tooling.

## License

Original project code is licensed under [Apache-2.0](LICENSE). Third-party
components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
