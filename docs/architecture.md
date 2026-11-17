# Compiler structure

This project follows the organization already present in the pinned MLIR source:
`mlir/lib/Transforms/CMakeLists.txt` groups transformations into a component,
`Canonicalizer.cpp` separates pass execution from pattern population, and
`mlir/include/mlir/Transforms/Passes.h` exposes pass factories. These local sources
were consulted during the cleanup; no IREE implementation or dependency was added.

## Components

| Component | Responsibility |
| --- | --- |
| `vbx_onnx_frontend` | Import/export ONNX, apply guarded patterns using the MLIR pass manager |
| `vbx_vectorblox_target` | Import/export SDK metadata, apply layout plans to actual tensor SSA |
| `vbx_rewrite` | Aggregate public library interface |
| `vbx-rewrite` | CLI parsing, atomic output and timing reports |

`Frontend/ONNX/ImportExport.cpp` owns protobuf conversion and graph bookkeeping.
`Passes.cpp` assembles patterns and implements the pass; `ChannelMean.cpp`,
`Legalize.cpp` and `Unfold.cpp` contain matching/lowering rules. `Passes.h`
declares the factory and `Patterns.h` declares pattern population functions. The internal `Bridge.h` carries
source metadata required by those rules. Public entry points are declared in
`include/vbx_rewrite/engine.h`.

`Target/VectorBlox/Graph.cpp` owns the tensor SSA representation and SDK metadata
round-trip. Its `Passes.cpp` runs guarded inverse-regroup lowering and upstream
MLIR canonicalization of proven inverse views, then layout legalization.
`Layout.cpp` validates shapes and quantization, plans a legal layout,
and applies it through MLIR PatternRewriter. Static tensor IDs are retained for
SDK interchange; dependencies are MLIR operands/results.

`tools/sdk_serialization.py` compacts unused tensor slots when handing rewritten
graphs to TFLite. It remaps public IO, signatures, optional operands and
intermediate references without changing operator order or buffer IDs. Empty
unused slots are insufficient: TensorFlow can enumerate them and then fail when
the SDK requests their unallocated data.

## Boundaries

- Matching and safety checks precede mutation. Unsupported cases remain unchanged.
- PatternRewriter owns operation replacement; MLIR owns the greedy worklist.
- JSON/protobuf adapters own serialization, not compiler CLI code or tests.
- Generic operations do not claim effect-free semantics. Broad CSE or
  canonicalization is not added without an operation contract.
- Tests and optional SDK compilation are separate from the native rewrite path.
- No model-name dispatch is embedded in C++ patterns. The retained rules are
  general guarded transformations with explicit shape, type and metadata constraints.
