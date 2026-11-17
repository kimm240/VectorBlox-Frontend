#include "Graph.h"
#include "mlir/IR/Verifier.h"
#include <stdexcept>
namespace vbx {
namespace {
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::vector<int64_t> ids(const llvm::json::Object &o, llvm::StringRef key) {
  std::vector<int64_t> v;
  if (auto *a = o.getArray(key))
    for (auto &x : *a) {
      auto i = x.getAsInteger();
      check(bool(i), "Invalid SDK tensor index");
      v.push_back(*i);
    }
  return v;
}
llvm::json::Array array(const std::vector<int64_t> &v) {
  llvm::json::Array a;
  for (auto x : v)
    a.push_back(x);
  return a;
}
} // namespace
llvm::json::Object &SdkSsaGraph::tensor(int64_t id) {
  check(id >= 0 && size_t(id) < tensors->size(),
        "SDK SSA tensor index out of range");
  auto *t = (*tensors)[id].getAsObject();
  check(t, "Invalid SDK SSA tensor record");
  return *t;
}
mlir::Type SdkSsaGraph::type(int64_t id) {
  if (id == -1)
    return mlir::NoneType::get(&context);
  auto &t = tensor(id);
  auto name = t.getString("type").value_or("");
  mlir::Type element;
  if (name == "FLOAT32")
    element = mlir::Float32Type::get(&context);
  else if (name == "FLOAT16")
    element = mlir::Float16Type::get(&context);
  else if (name == "FLOAT64")
    element = mlir::Float64Type::get(&context);
  else {
    unsigned bits = name == "BOOL"                          ? 1
                    : (name == "INT8" || name == "UINT8")   ? 8
                    : (name == "INT16" || name == "UINT16") ? 16
                    : (name == "INT32" || name == "UINT32") ? 32
                    : (name == "INT64" || name == "UINT64") ? 64
                                                            : 0;
    if (bits)
      element = mlir::IntegerType::get(&context, bits,
                                       name.starts_with("UINT")
                                           ? mlir::IntegerType::Unsigned
                                           : mlir::IntegerType::Signless);
  }
  if (!element)
    return mlir::OpaqueType::get(mlir::StringAttr::get(&context, "vbx_sdk"),
                                 name);
  auto *shape = t.getArray("shape");
  if (!shape)
    return mlir::UnrankedTensorType::get(element);
  llvm::SmallVector<int64_t> dims;
  for (auto &d : *shape) {
    auto n = d.getAsInteger();
    check(n && *n >= -1, "Invalid SDK SSA dimension");
    dims.push_back(*n);
  }
  return mlir::RankedTensorType::get(dims, element);
}
SdkSsaGraph::SdkSsaGraph(llvm::json::Object &r) : root(r) {
  auto *gs = root.getArray("subgraphs");
  check(gs && gs->size() == 1, "SDK SSA requires one subgraph");
  graph = (*gs)[0].getAsObject();
  check(graph, "Invalid SDK SSA subgraph");
  operators = graph->getArray("operators");
  tensors = graph->getArray("tensors");
  buffers = root.getArray("buffers");
  codes = root.getArray("operator_codes");
  check(operators && tensors && buffers && codes, "Missing SDK SSA tables");
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder b(&context);
  module = mlir::ModuleOp::create(b.getUnknownLoc());
  auto inputs = ids(*graph, "inputs"), outputs = ids(*graph, "outputs");
  for (auto id : inputs)
    exposed.insert(id);
  for (auto id : outputs)
    exposed.insert(id);
  if (auto *signatures = root.getArray("signature_defs"))
    for (auto &s : *signatures) {
      auto *sig = s.getAsObject();
      check(sig, "Invalid SDK signature");
      for (auto key : {"inputs", "outputs"})
        if (auto *a = sig->getArray(key))
          for (auto &entry : *a) {
            auto *e = entry.getAsObject();
            check(e, "Invalid SDK signature entry");
            auto id = e->getInteger("tensor_index");
            check(bool(id), "Missing SDK signature tensor");
            exposed.insert(*id);
          }
    }
  llvm::SmallVector<mlir::Type> inTypes, outTypes;
  for (auto id : inputs)
    inTypes.push_back(type(id));
  for (auto id : outputs)
    outTypes.push_back(type(id));
  function = mlir::func::FuncOp::create(b.getUnknownLoc(), "sdk_graph",
                                        b.getFunctionType(inTypes, outTypes));
  module->push_back(function);
  auto *block = function.addEntryBlock();
  b.setInsertionPointToStart(block);
  for (size_t i = 0; i < inputs.size(); ++i) {
    check(!values.count(inputs[i]), "Duplicate SDK graph input");
    values[inputs[i]] = block->getArgument(i);
    tensorIds[block->getArgument(i)] = inputs[i];
  }
  std::set<int64_t> produced;
  for (auto &v : *operators) {
    auto *o = v.getAsObject();
    check(o, "Invalid SDK SSA operator");
    for (auto id : ids(*o, "outputs")) {
      tensor(id);
      check(produced.insert(id).second && !values.count(id),
            "Multiple SDK SSA tensor definitions");
    }
    auto in = ids(*o, "inputs");
    if (auto *flags = o->getArray("mutating_variable_inputs")) {
      check(flags->empty() || flags->size() == in.size(),
            "Invalid SDK mutation flags");
      for (size_t i = 0; i < flags->size(); ++i)
        if ((*flags)[i].getAsBoolean().value_or(true))
          mutated.insert(in[i]);
    }
  }
  auto leaf = [&](int64_t id) {
    mlir::OperationState state(b.getUnknownLoc(),
                               id == -1 ? "vbx_sdk.none" : "vbx_sdk.source");
    state.addTypes(type(id));
    state.addAttribute("tensor_id", b.getI64IntegerAttr(id));
    auto *op = b.create(state);
    values[id] = op->getResult(0);
    tensorIds[op->getResult(0)] = id;
  };
  for (size_t i = 0; i < operators->size(); ++i) {
    auto &o = *(*operators)[i].getAsObject();
    auto in = ids(o, "inputs"), out = ids(o, "outputs");
    llvm::SmallVector<mlir::Value> operands;
    llvm::SmallVector<mlir::Type> types;
    for (auto id : in) {
      if (!values.count(id)) {
        check(!produced.count(id), "SDK SSA use precedes its definition");
        leaf(id);
      }
      operands.push_back(values.at(id));
    }
    for (auto id : out)
      types.push_back(type(id));
    auto ci = o.getInteger("opcode_index");
    check(ci && *ci >= 0 && size_t(*ci) < codes->size(),
          "Invalid SDK SSA opcode");
    auto *code = (*codes)[*ci].getAsObject();
    check(code, "Invalid SDK SSA opcode record");
    mlir::OperationState state(b.getUnknownLoc(), "vbx_sdk.node");
    state.addOperands(operands);
    state.addTypes(types);
    state.addAttribute(
        "builtin",
        b.getStringAttr(code->getString("builtin_code").value_or("CUSTOM")));
    auto *op = b.create(state);
    original[op] = i;
    for (size_t j = 0; j < out.size(); ++j) {
      values[out[j]] = op->getResult(j);
      tensorIds[op->getResult(j)] = out[j];
    }
  }
  llvm::SmallVector<mlir::Value> returns;
  for (auto id : outputs) {
    if (!values.count(id)) {
      check(!produced.count(id), "Missing SDK SSA output");
      leaf(id);
    }
    returns.push_back(values.at(id));
  }
  b.create<mlir::func::ReturnOp>(b.getUnknownLoc(), returns);
  check(mlir::succeeded(mlir::verify(*module)),
        "Invalid imported SDK SSA graph");
}
llvm::json::Object &SdkSsaGraph::record(mlir::Operation *op) {
  auto it = original.find(op);
  if (it != original.end())
    return *(*operators)[it->second].getAsObject();
  auto made = generated.find(op);
  check(made != generated.end(), "Missing SDK SSA node record");
  return made->second;
}
int64_t SdkSsaGraph::internConstantBuffer(llvm::ArrayRef<int64_t> bytes) {
  std::string payload;
  payload.reserve(bytes.size());
  for (auto byte : bytes) {
    check(byte >= 0 && byte <= 255, "Invalid generated SDK constant byte");
    payload.push_back(static_cast<char>(byte));
  }
  // MLIR context uniquing compares the complete byte string (including NULs).
  // This shares storage only; it is not CSE of SDK operations or tensor values.
  auto key = mlir::StringAttr::get(&context, payload);
  auto found = constantBuffers.find(key);
  if (found != constantBuffers.end())
    return found->second;
  auto index = int64_t(buffers->size());
  llvm::json::Array data;
  for (auto byte : bytes)
    data.push_back(byte);
  buffers->push_back(llvm::json::Object{{"data", std::move(data)}});
  constantBuffers.try_emplace(key, index);
  return index;
}
mlir::Value SdkSsaGraph::addConstant(mlir::PatternRewriter &rw, int64_t id) {
  mlir::OperationState state(rw.getUnknownLoc(), "vbx_sdk.source");
  state.addTypes(type(id));
  state.addAttribute("tensor_id", rw.getI64IntegerAttr(id));
  auto *op = rw.create(state);
  auto value = op->getResult(0);
  tensorIds[value] = id;
  values[id] = value;
  return value;
}
mlir::Operation *SdkSsaGraph::addNode(mlir::PatternRewriter &rw,
                                      llvm::json::Object rec,
                                      mlir::ValueRange inputs, int64_t output) {
  return addMultiResultNode(rw, std::move(rec), inputs,
                            llvm::ArrayRef<int64_t>(output));
}
mlir::Operation *
SdkSsaGraph::addMultiResultNode(mlir::PatternRewriter &rw,
                                llvm::json::Object rec, mlir::ValueRange inputs,
                                llvm::ArrayRef<int64_t> outputs) {
  check(!outputs.empty(), "SDK SSA generated node requires results");
  auto ci = rec.getInteger("opcode_index").value();
  auto *code = (*codes)[ci].getAsObject();
  mlir::OperationState state(rw.getUnknownLoc(), "vbx_sdk.node");
  state.addOperands(inputs);
  for (auto output : outputs)
    state.addTypes(type(output));
  state.addAttribute(
      "builtin",
      rw.getStringAttr(code->getString("builtin_code").value_or("CUSTOM")));
  auto *op = rw.create(state);
  generated.try_emplace(op, std::move(rec));
  for (size_t i = 0; i < outputs.size(); ++i) {
    tensorIds[op->getResult(i)] = outputs[i];
    values[outputs[i]] = op->getResult(i);
  }
  return op;
}

// Erased MLIR operation addresses may be recycled by the allocator. Remove
// bookkeeping before erasure so a later generated node cannot inherit metadata.
void SdkSsaGraph::forgetNode(mlir::Operation *op) {
  original.erase(op);
  generated.erase(op);
  for (auto value : op->getResults()) {
    auto it = tensorIds.find(value);
    if (it == tensorIds.end())
      continue;
    auto current = values.find(it->second);
    if (current != values.end() && current->second == value)
      values.erase(current);
    tensorIds.erase(it);
  }
}
void SdkSsaGraph::exportOperators() {
  check(mlir::succeeded(mlir::verify(*module)),
        "Invalid rewritten SDK SSA graph");
  llvm::json::Array result;
  for (auto &op : function.getBody().front()) {
    if (op.getName().getStringRef() != "vbx_sdk.node")
      continue;
    auto rec = std::move(record(&op));
    std::vector<int64_t> in, out;
    for (auto v : op.getOperands()) {
      check(tensorIds.count(v), "Unmapped SDK SSA operand");
      in.push_back(tensorIds.lookup(v));
    }
    for (auto v : op.getResults()) {
      check(tensorIds.count(v), "Unmapped SDK SSA result");
      out.push_back(tensorIds.lookup(v));
    }
    rec["inputs"] = array(in);
    rec["outputs"] = array(out);
    result.push_back(std::move(rec));
  }
  *operators = std::move(result);
}
} // namespace vbx
