# Third-party notices

Original project code is distributed under the root Apache-2.0 LICENSE. Third-party components retain their own licenses and notices.

- LLVM and MLIR 20.1.8, including CMake and third-party build support: https://github.com/llvm/llvm-project/tree/llvmorg-20.1.8 . Primarily Apache-2.0 WITH LLVM-exception; additional component notices apply. Bootstrap preserves license files in the extracted source and original archives. We link upstream implementations rather than copying and maintaining a fork of PatternRewriter.
- Protocol Buffers 3.20.3: https://github.com/protocolbuffers/protobuf/tree/v3.20.3 . BSD 3-Clause; license included in its cached archive and extracted source.
- ONNX 1.16.1 schema: https://github.com/onnx/onnx/blob/v1.16.1/onnx/onnx.proto . Apache-2.0, license at [ONNX license](../third_party/onnx/LICENSE). The checked-in schema was obtained from ONNX 1.16.1. CMake generates a build-local copy changing `optimize_for` from LITE_RUNTIME to SPEED for recursive protobuf reflection; the wire schema is unchanged.

Upstream MLIR facilities used by this project:

- `mlir/include/mlir/IR/PatternMatch.h`
- `mlir/lib/Rewrite/PatternApplicator.cpp`
- `mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp`
- MLIR IR, Func dialect, Pass and TransformUtils libraries and their required dependencies.

Pinned download URLs and SHA256 digests are in [dependency lockfile](../third_party/dependencies.lock.json). If distributing archives, sources or binaries, retain the corresponding third-party license and notice files. Optional Python validation packages are separate dependencies with their own licenses. Model weights are not included in the source distribution.
