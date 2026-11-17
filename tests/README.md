# Standalone tests

Tests never run inside normal compilation. Enable the C++ core suite explicitly:

```sh
cmake -S . -B build -G Ninja -DVBX_BUILD_TESTS=ON
cmake --build build --target vbx-rewrite vbx-rewrite-tests -j 4
ctest --test-dir build --output-on-failure
```

Install `requirements-test.txt` in a test environment for the Python suites.
Run from the project root:

```sh
python tests/validate_numerical_comparator.py
python tests/validate_onnx.py
python tests/validate_unfold.py
python tests/validate_legalizations.py
python tests/validate_sdk_layout.py
python tests/validate_sdk_layout_ssa.py
python tests/validate_sdk_serialization.py
python tests/validate_sdk_inverse_regroup.py
python tests/validate_sdk_inverse_views.py
```

| Suite | Coverage |
| --- | --- |
| `validate_numerical_comparator.py` | End-to-end comparator controls: identity accepted; wrong values/names/shapes and nonfinite outputs rejected |
| `rewrite_test.cpp`, `validate_onnx.py` | Channel mean, input legality, public outputs, numerical checks |
| `validate_unfold.py` | Static unfold patterns and rejection guards |
| `validate_legalizations.py` | Attention, GELU, LayerNorm, Gather, linear, ConvTranspose and Slice rules; numerical equivalence |
| `validate_sdk_layout.py` | Exact element order, metadata and mutation guards, idempotence, budget |
| `validate_sdk_layout_ssa.py` | Dependent layout rewrites, true SSA edges, output IDs and atomic budget failure |
| `validate_sdk_serialization.py` | Tensor compaction preserves public IO, signatures, optional operands, intermediates and buffers |
| `validate_sdk_inverse_regroup.py`, `validate_sdk_inverse_views.py` | Produced tensor regrouping and guarded upstream MLIR inverse-view canonicalization |
| `validate_dependency_patches.py` | Pinned dependency patch application |

Compare full ONNX models independently:

```sh
python tests/compare_onnx.py before.onnx after.onnx --samples 2 --edge-cases --report numerical.json
```

The comparison matches named outputs, disables ONNX Runtime graph fusions,
and checks absolute/relative tolerances. Synthetic cases do not establish task
accuracy. Use `--inputs` for model-domain inputs. Supply your own before/after
models; test fixtures are generated locally and no benchmark model downloads
are required.

`compare_tflite_models.py` provides a separate sampled integer-output comparison
for serialized SDK graph rewrites. It requires TensorFlow in an SDK-compatible
Python environment in addition to the core ONNX test requirements. VNNX runtime
and board performance tests are outside this repository's legalization scope.

GitHub Actions runs the standalone suites in `.github/workflows/onnx-numerical.yml`.
It compares generated ONNX fixtures before/after the native rewrite, with ORT
graph optimizations disabled; pretrained model weights and VNNX/board tests are
not part of CI. General legalization uses `atol=rtol=1e-4`, channel mean uses
`3e-5`, and unfold requires exact equality.
