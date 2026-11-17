#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <set>
namespace vbx {
// Lossless SDK metadata stays in the JSON tables. Actual tensor dependencies
// are SSA operands/results; source indices are bookkeeping, not fake IR edges.
// Generic nodes stay conservative. Dedicated stages register guarded subsets
// with explicit semantics before using standard CSE/canonicalization.
class SdkSsaGraph {
public:
  llvm::json::Object &root;
  llvm::json::Object *graph;
  llvm::json::Array *operators, *tensors, *buffers, *codes;
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  llvm::DenseMap<mlir::Operation *, size_t> original;
  llvm::DenseMap<mlir::Operation *, llvm::json::Object> generated;
  llvm::DenseMap<mlir::Value, int64_t> tensorIds;
  std::map<int64_t, mlir::Value> values;
  std::set<int64_t> exposed, mutated;
  // Only newly generated immutable payloads participate. Tensor identities and
  // their quantization remain separate, including private SDK lookup chains.
  llvm::DenseMap<mlir::StringAttr, int64_t> constantBuffers;
  explicit SdkSsaGraph(llvm::json::Object &root);
  mlir::Type type(int64_t id);
  llvm::json::Object &tensor(int64_t id);
  llvm::json::Object &record(mlir::Operation *op);
  mlir::Value addConstant(mlir::PatternRewriter &rw, int64_t id);
  int64_t internConstantBuffer(llvm::ArrayRef<int64_t> bytes);
  mlir::Operation *addNode(mlir::PatternRewriter &rw, llvm::json::Object record,
                           mlir::ValueRange inputs, int64_t output);
  // Preserve a separate tensor ID and SSA value for every generated result.
  mlir::Operation *addMultiResultNode(mlir::PatternRewriter &rw,
                                      llvm::json::Object record,
                                      mlir::ValueRange inputs,
                                      llvm::ArrayRef<int64_t> outputs);
  void forgetNode(mlir::Operation *op);
  // Finalize this bridge: transfers operator records into the root JSON.
  // The module may still be printed, but record() must not be used afterward.
  void exportOperators();
};
} // namespace vbx
