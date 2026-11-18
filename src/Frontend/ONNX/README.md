# ONNX legalization

The `vbx-legalize-onnx` MLIR pass rewrites selected ONNX patterns into forms
that VectorBlox SDK 3.1 can handle. These are patterns within one pass, not
separate passes for each operator. Matching uses graph structure and tensor
metadata, never model names.

## Implemented patterns

| Source | Original limitation / motivation | Rewrite |
| --- | --- | --- |
| [ChannelMean.cpp](ChannelMean.cpp) | Channel-wise mean reduction in XFeat needs a supported lowering | Static NCHW FP32 `ReduceMean`, channel axis with `keepdims=1`, becomes a 1×1 Conv with averaging weights |
| [Unfold.cpp](Unfold.cpp) | Exported static unfold chains are not directly supported | Recognized reshape/transpose/slice patterns become a one-hot strided Conv |
| [Legalize.cpp](Legalize.cpp): MatMul / Gemm | Linear and attention tensor layouts need SDK-compatible lowering | Guarded Conv-based linear lowering and attention decomposition |
| LayerNormalization | Rank-changing normalization fusion and large intermediate broadcasts caused SDK allocation failures | Decompose normalization into supported arithmetic and Conv reductions; tile independent rows and place the square after layout conversion |
| Softmax / GELU | Large exported activation patterns stress SDK LUT handling | Tile independent rows; preserve the recognized GELU chain's constants and arithmetic order |
| Scalar Mul | Scalar scaling after QKV layout changes may become a large SDK LUT | Lower eligible scalar multiplication to depthwise Conv |
| Gather | Selected constant-index gathers need supported forms | Static slicing and layout rewrites for eligible indices and axes |
| High-rank Slice | Selected high-rank slices need simpler layouts | Reshape and slice eligible static dimensions |
| ConvTranspose | Selected transposed-convolution forms need decomposition | Decompose the supported form into simpler operators; other forms remain unchanged |

These descriptions explain why the patterns exist; they do not mean every
instance of the original operator is unsupported or every variant is rewritten.

## Conditions and safety

- Shapes, dtypes, opsets, attributes, metadata and cost limits must match each
  pattern. Most arithmetic patterns require static FP32 tensors.
- Patterns affecting order-sensitive consumers have additional guards.
- Unsupported cases remain unchanged; skip reasons appear in the rewrite report.
- Only the main graph is rewritten. Nested graphs and functions are preserved.
- `--max-channels` and `--max-rewrites` bound channel policy and rewrite count.

[Passes.cpp](Passes.cpp) registers the patterns and runs upstream MLIR's greedy
rewrite driver. [ImportExport.cpp](ImportExport.cpp) handles ONNX import/export;
[Bridge.h](Bridge.h) carries source graph metadata through the MLIR representation.

See [legalization tests](../../../tests/legalization/README.md) for per-pattern
regressions and full-model conversion/numerical checks.
