#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "vbx_rewrite/engine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <set>

namespace vbx {
struct GeneratedWeight {
  std::string name;
  int64_t channels;
  int64_t block = 1;
};
struct Bridge {
  onnx::ModelProto &model;
  const Options &options;
  Stats stats;
  int64_t opset = 0;
  mlir::func::FuncOp function;
  llvm::SmallVector<mlir::Operation *> candidates;
  llvm::DenseMap<mlir::Value, const onnx::TensorProto *> constants;
  llvm::StringSet<> names;
  llvm::DenseMap<int64_t, uint64_t> nameCounters;
  std::vector<GeneratedWeight> generated;
  std::set<int64_t> removed;
  std::set<int64_t> orderSensitive;
  std::map<int64_t, std::vector<onnx::NodeProto>> replacements;
  std::vector<onnx::TensorProto> tensors;
  std::string uniqueWeightName(int64_t node_id);
};
} // namespace vbx
