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
python tests/validate_dependency_patches.py
```

| Suite | Coverage |
| --- | --- |
| `validate_numerical_comparator.py` | End-to-end comparator controls: identity accepted; wrong values/names/shapes and nonfinite outputs rejected |
| `rewrite_test.cpp`, `validate_onnx.py` | Channel mean, input legality, public outputs, numerical checks |
| `validate_unfold.py` | Static unfold patterns and rejection guards |
| `validate_legalizations.py` | Attention, GELU, LayerNorm, Gather, linear, ConvTranspose and Slice rules; numerical equivalence |
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

GitHub Actions runs the standalone suites in `.github/workflows/onnx-numerical.yml`.
It compares generated ONNX fixtures before/after the native rewrite, with ORT
graph optimizations disabled; pretrained model weights and VNNX/board tests are
not part of CI. General legalization uses `atol=rtol=1e-4`, channel mean uses
`3e-5`, and unfold requires exact equality.
