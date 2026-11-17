#pragma once
#include "onnx.pb.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace vbx {
struct Options {
  int64_t max_channels = 4096;
  int64_t max_rewrites = 1000000;
};
struct Stats {
  size_t nodes = 0, candidates = 0, attempts = 0, rewrites = 0;
  std::map<std::string, size_t> skipped;
  double setup_ms = 0, import_ms = 0, rewrite_ms = 0, export_ms = 0;
};
// Uses upstream MLIR's pass manager, pattern rewriter and greedy driver.
// Main graph only; nested graphs/functions are preserved, not rewritten.
Stats rewriteModel(onnx::ModelProto &model, const Options &options = {},
                   std::string *mlir_dump = nullptr);
bool hasExternalData(const onnx::ModelProto &model);
// Layout-only legalization of the SDK's flatc TFLite JSON representation.
// Preserves quantization and uses the same upstream MLIR rewrite driver.
Stats rewriteSdkJson(std::string &json, const Options &options = {},
                     std::string *mlir_dump = nullptr);
} // namespace vbx
