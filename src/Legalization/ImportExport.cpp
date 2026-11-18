#include "Bridge.h"
#include "Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstring>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <stdexcept>

namespace vbx {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
mlir::Type elementType(mlir::Builder &b, int type) {
  switch (type) {
  case onnx::TensorProto::FLOAT:
    return b.getF32Type();
  case onnx::TensorProto::DOUBLE:
    return b.getF64Type();
  case onnx::TensorProto::FLOAT16:
    return b.getF16Type();
  case onnx::TensorProto::BFLOAT16:
    return b.getBF16Type();
  case onnx::TensorProto::INT64:
  case onnx::TensorProto::UINT64:
    return b.getI64Type();
  case onnx::TensorProto::INT32:
  case onnx::TensorProto::UINT32:
    return b.getI32Type();
  case onnx::TensorProto::INT16:
  case onnx::TensorProto::UINT16:
    return b.getI16Type();
  case onnx::TensorProto::INT8:
  case onnx::TensorProto::UINT8:
    return b.getI8Type();
  case onnx::TensorProto::BOOL:
    return b.getI1Type();
  default:
    return {};
  }
}
mlir::Type valueType(mlir::Builder &b, const onnx::ValueInfoProto *info) {
  auto opaque = mlir::OpaqueType::get(b.getStringAttr("vbx_onnx"), "value");
  if (!info || !info->type().has_tensor_type())
    return opaque;
  auto &tensor = info->type().tensor_type();
  auto element = elementType(b, tensor.elem_type());
  if (!element)
    return opaque;
  if (!tensor.has_shape())
    return mlir::UnrankedTensorType::get(element);
  llvm::SmallVector<int64_t> shape;
  for (auto &d : tensor.shape().dim()) {
    if (d.has_dim_value() && d.dim_value() < 0)
      throw std::runtime_error("Negative ONNX dimension");
    shape.push_back(d.has_dim_value() ? d.dim_value()
                                      : mlir::ShapedType::kDynamic);
  }
  return mlir::RankedTensorType::get(shape, element);
}
mlir::Type tensorType(mlir::Builder &b, const onnx::TensorProto &tensor) {
  auto element = elementType(b, tensor.data_type());
  if (!element)
    return mlir::OpaqueType::get(b.getStringAttr("vbx_onnx"), "value");
  llvm::SmallVector<int64_t> dims(tensor.dims().begin(), tensor.dims().end());
  for (auto d : dims)
    if (d < 0)
      throw std::runtime_error("Negative initializer dimension");
  return mlir::RankedTensorType::get(dims, element);
}
mlir::Operation *proxy(mlir::OpBuilder &b, llvm::StringRef name,
                       mlir::TypeRange types, mlir::ValueRange inputs = {},
                       int64_t source = -1) {
  mlir::OperationState state(b.getUnknownLoc(), name);
  state.addTypes(types);
  state.addOperands(inputs);
  if (source >= 0)
    state.addAttribute("vbx.source", b.getI64IntegerAttr(source));
  return b.create(state);
}
void importGraph(Bridge &bridge, mlir::ModuleOp module) {
  auto &graph = bridge.model.graph();
  if (graph.sparse_initializer_size())
    throw std::runtime_error(
        "Sparse initializers not supported by this adapter");
  // A tiny floating-point perturbation can reorder tied TopK/ArgMax results.
  // Protect their transitive data dependencies from arithmetic substitution.
  llvm::StringSet<> sensitive;
  for (int i = graph.node_size() - 1; i >= 0; --i) {
    auto &n = graph.node(i);
    bool selected =
        n.domain().empty() &&
        (n.op_type() == "TopK" || n.op_type() == "ArgMax" ||
         n.op_type() == "ArgMin" || n.op_type() == "NonMaxSuppression");
    bool depends = selected;
    for (auto &output : n.output())
      depends |= sensitive.count(output) != 0;
    if (depends) {
      bridge.orderSensitive.insert(i);
      for (auto &input : n.input())
        if (!input.empty())
          sensitive.insert(input);
    }
  }
  mlir::OpBuilder b(module.getContext());
  llvm::StringMap<const onnx::ValueInfoProto *> infos;
  for (auto &v : graph.value_info())
    infos[v.name()] = &v;
  for (auto &v : graph.output())
    infos[v.name()] = &v;
  for (auto &v : graph.input())
    infos[v.name()] = &v;
  auto typeFor = [&](const std::string &name) {
    return valueType(b, infos.lookup(name));
  };
  llvm::SmallVector<mlir::Type> inputs, outputs;
  for (auto &v : graph.input())
    inputs.push_back(typeFor(v.name()));
  for (auto &v : graph.output())
    outputs.push_back(typeFor(v.name()));
  bridge.function = mlir::func::FuncOp::create(
      b.getUnknownLoc(), "main", b.getFunctionType(inputs, outputs));
  module.push_back(bridge.function);
  auto *block = bridge.function.addEntryBlock();
  b.setInsertionPointToStart(block);
  llvm::StringMap<mlir::Value> values(
      graph.node_size() * 2 + graph.initializer_size() + graph.input_size());
  auto bind = [&](const std::string &name, mlir::Value value) {
    if (name.empty() || !values.try_emplace(name, value).second)
      throw std::runtime_error("Empty or duplicate tensor definition: " + name);
    bridge.names.insert(name);
  };
  for (int i = 0; i < graph.input_size(); ++i)
    bind(graph.input(i).name(), block->getArgument(i));
  llvm::StringSet<> initializerNames;
  for (auto &t : graph.initializer()) {
    if (!initializerNames.insert(t.name()).second)
      throw std::runtime_error("Duplicate initializer: " + t.name());
    // An initializer which is also a graph input is overridable, not a
    // constant.
    if (values.count(t.name()))
      continue;
    auto *op = proxy(b, "vbx_onnx.weight", {tensorType(b, t)});
    bind(t.name(), op->getResult(0));
    bridge.constants[op->getResult(0)] = &t;
  }
  mlir::Value none;
  for (int i = 0; i < graph.node_size(); ++i) {
    auto &node = graph.node(i);
    llvm::SmallVector<mlir::Value> operands;
    llvm::SmallVector<mlir::Type> types;
    for (auto &name : node.input()) {
      if (name.empty()) {
        if (!none)
          none = proxy(b, "vbx_onnx.none", {b.getNoneType()})->getResult(0);
        operands.push_back(none);
      } else {
        auto it = values.find(name);
        if (it == values.end())
          throw std::runtime_error("Undefined or non-topological input: " +
                                   name);
        operands.push_back(it->second);
      }
    }
    for (auto &name : node.output())
      if (!name.empty())
        types.push_back(typeFor(name));
    bool mean = node.domain().empty() && node.op_type() == "ReduceMean";
    bool reshape = node.domain().empty() && node.op_type() == "Reshape";
    bool legal =
        node.domain().empty() &&
        (node.op_type() == "MatMul" || node.op_type() == "Gemm" ||
         node.op_type() == "ConvTranspose" || node.op_type() == "Slice" ||
         node.op_type() == "Gather" || node.op_type() == "LayerNormalization" ||
         node.op_type() == "Softmax" || node.op_type() == "Mul" || node.op_type() == "Add");
    std::string opname = mean      ? "vbx_onnx.ReduceMean"
                         : reshape ? "vbx_onnx.Reshape"
                         : legal   ? "vbx_onnx." + node.op_type()
                                   : "vbx_onnx.node";
    auto *op = proxy(b, opname, types, operands, i);
    unsigned result = 0;
    for (auto &name : node.output())
      if (!name.empty())
        bind(name, op->getResult(result++));
    if (mean || reshape || legal)
      bridge.candidates.push_back(op);
    if (node.domain().empty() && node.op_type() == "Constant" &&
        op->getNumResults() == 1)
      for (auto &a : node.attribute())
        if (a.name() == "value" && a.type() == onnx::AttributeProto::TENSOR &&
            a.has_t())
          bridge.constants[op->getResult(0)] = &a.t();
  }
  llvm::SmallVector<mlir::Value> returned;
  for (auto &output : graph.output()) {
    auto it = values.find(output.name());
    if (it == values.end())
      throw std::runtime_error("Undefined graph output: " + output.name());
    returned.push_back(it->second);
  }
  b.create<mlir::func::ReturnOp>(b.getUnknownLoc(), returned);
}
void exportChanges(Bridge &bridge) {
  auto *graph = bridge.model.mutable_graph();
  for (auto &weight : bridge.generated) {
    auto *t = graph->add_initializer();
    t->set_name(weight.name);
    t->set_data_type(onnx::TensorProto::FLOAT);
    for (auto d : {weight.block * weight.block, weight.channels, weight.block,
                   weight.block})
      t->add_dims(d);
    // Protobuf packed float data is portable across host byte orders.
    t->mutable_float_data()->Reserve(static_cast<int>(weight.channels));
    const float scale = 1.0f / static_cast<float>(weight.channels);
    if (weight.block == 1) {
      for (int64_t c = 0; c < weight.channels; ++c)
        t->add_float_data(scale);
    } else {
      auto area = weight.block * weight.block;
      for (int64_t o = 0; o < area; ++o)
        for (int64_t i = 0; i < area; ++i)
          t->add_float_data(o == i ? 1.0f : 0.0f);
    }
  }
  for (auto &op : bridge.function.getBody().front()) {
    auto rewritten = op.getAttrOfType<mlir::StringAttr>("vbx.generated_weight");
    if (!rewritten)
      continue;
    int64_t id = op.getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt();
    auto *node = graph->mutable_node(static_cast<int>(id));
    std::string input = node->input(0);
    if (auto source = op.getAttrOfType<mlir::StringAttr>("vbx.input"))
      input = source.getValue().str();
    auto blockAttr = op.getAttrOfType<mlir::IntegerAttr>("vbx.block");
    int64_t block = blockAttr ? blockAttr.getInt() : 1;
    node->set_op_type("Conv");
    node->clear_domain();
    node->clear_attribute();
    node->clear_input();
    node->add_input(std::move(input));
    node->add_input(rewritten.getValue().str());
    // onnx2tf 1.22.3 requires explicit kernel rank instead of inferring it
    // from the weight shape, even though ONNX permits omitting kernel_shape.
    auto *kernel = node->add_attribute();
    kernel->set_name("kernel_shape");
    kernel->set_type(onnx::AttributeProto::INTS);
    kernel->add_ints(block);
    kernel->add_ints(block);
    if (block > 1) {
      auto *stride = node->add_attribute();
      stride->set_name("strides");
      stride->set_type(onnx::AttributeProto::INTS);
      stride->add_ints(block);
      stride->add_ints(block);
    }
    // Names, outputs, doc strings, metadata and untouched nodes remain
    // in-place.
  }
  for (auto &tensor : bridge.tensors)
    graph->add_initializer()->Swap(&tensor);
  if (!bridge.removed.empty() || !bridge.replacements.empty()) {
    llvm::StringSet<> deletedNames;
    for (auto id : bridge.removed)
      for (auto &name : graph->node(id).output())
        deletedNames.insert(name);
    google::protobuf::RepeatedPtrField<onnx::NodeProto> nodes;
    for (int i = 0; i < graph->node_size(); ++i) {
      if (bridge.removed.count(i))
        continue;
      auto replacement = bridge.replacements.find(i);
      if (replacement == bridge.replacements.end())
        nodes.Add()->Swap(graph->mutable_node(i));
      else
        for (auto &n : replacement->second)
          nodes.Add()->Swap(&n);
    }
    graph->mutable_node()->Swap(&nodes);
    for (int i = graph->value_info_size() - 1; i >= 0; --i)
      if (deletedNames.count(graph->value_info(i).name()))
        graph->mutable_value_info()->DeleteSubrange(i, 1);
  }
}
bool containsExternal(const google::protobuf::Message &message) {
  auto *reflection = message.GetReflection();
  if (message.GetDescriptor() == onnx::TensorProto::descriptor()) {
    auto *field = message.GetDescriptor()->FindFieldByName("data_location");
    if (reflection->GetEnumValue(message, field) == onnx::TensorProto::EXTERNAL)
      return true;
    if (reflection->FieldSize(
            message, message.GetDescriptor()->FindFieldByName("external_data")))
      return true;
  }
  std::vector<const google::protobuf::FieldDescriptor *> fields;
  reflection->ListFields(message, &fields);
  for (auto *field : fields)
    if (field->cpp_type() ==
        google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      if (field->is_repeated()) {
        for (int i = 0; i < reflection->FieldSize(message, field); ++i)
          if (containsExternal(
                  reflection->GetRepeatedMessage(message, field, i)))
            return true;
      } else if (containsExternal(reflection->GetMessage(message, field)))
        return true;
    }
  return false;
}
} // namespace
std::string Bridge::uniqueWeightName(int64_t id) {
  std::string base = "__vbx_mean_" + std::to_string(id) + "_weight";
  auto &next = nameCounters[id];
  // The name set only grows. Resume the same deterministic collision search
  // instead of rescanning every earlier name for each emitted tensor.
  for (;;) {
    auto suffix = next++;
    auto name = suffix ? base + "_" + std::to_string(suffix - 1) : base;
    if (names.insert(name).second)
      return name;
  }
}
Stats rewriteModel(onnx::ModelProto &model, const Options &options,
                   std::string *dump) {
  auto setup = Clock::now();
  if (!model.has_graph())
    throw std::runtime_error("Missing ONNX graph");
  if (options.max_channels <= 0 || options.max_channels > 1048576 ||
      options.max_rewrites <= 0)
    throw std::runtime_error("Invalid resource limits");
  Bridge bridge{model, options};
  bridge.stats.nodes = model.graph().node_size();
  for (auto &opset : model.opset_import())
    if (opset.domain().empty()) {
      if (bridge.opset)
        throw std::runtime_error("Duplicate default opset import");
      bridge.opset = opset.version();
    }
  mlir::MLIRContext context;
  context.disableMultithreading();
  context.allowUnregisteredDialects();
  context.loadDialect<mlir::func::FuncDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  bridge.stats.setup_ms = elapsed(setup);
  auto time = Clock::now();
  importGraph(bridge, *module);
  if (mlir::failed(mlir::verify(*module)))
    throw std::runtime_error("Imported MLIR verification failed");
  bridge.stats.import_ms = elapsed(time);
  bridge.stats.candidates = bridge.candidates.size();
  time = Clock::now();
  mlir::PassManager pm(&context);
  pm.enableVerifier(true);
  pm.addPass(createLegalizePass(bridge));
  if (mlir::failed(pm.run(*module)))
    throw std::runtime_error("MLIR rewrite failed to converge or verify");
  bridge.stats.rewrite_ms = elapsed(time);
  if (dump) {
    llvm::raw_string_ostream stream(*dump);
    module->print(stream);
  }
  time = Clock::now();
  exportChanges(bridge);
  bridge.stats.export_ms = elapsed(time);
  return bridge.stats;
}
bool hasExternalData(const onnx::ModelProto &model) {
  return containsExternal(model);
}
} // namespace vbx
