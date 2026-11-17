#include "Graph.h"
#include "Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "vbx_rewrite/engine.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace vbx {
namespace {
using Shape = std::vector<int64_t>;
struct Replacement {
  size_t index;
  Shape permutation;
  Shape intermediate;
  Shape extra = {};
  std::vector<Shape> chain = {};
  int transposeStep = -1;
};
bool integers(const llvm::json::Array *a, Shape &out) {
  if (!a)
    return false;
  for (auto &v : *a) {
    auto i = v.getAsInteger();
    if (!i)
      return false;
    out.push_back(*i);
  }
  return true;
}
llvm::json::Array array(const Shape &values) {
  llvm::json::Array a;
  for (auto v : values)
    a.push_back(v);
  return a;
}
// Backend-specific legality plans. The plans retain JSON metadata; applying
// them below rewires actual tensor SSA, not a disconnected shape-only graph.
std::optional<Replacement> layoutPlan(size_t index, const Shape &in,
                                      const Shape &out) {
  if (out.size() == 3) {
    if ((in[1] == out[0] && in[2] == out[1] && in[3] == out[2]) || in[0] != 1 ||
        out[0] <= 1 || out[0] > in[1] || in[1] % out[0] ||
        in[1] / out[0] != out[1] || in[2] > 16777216 / in[3] ||
        in[2] * in[3] != out[2] || in[1] > 16777216 / out[2])
      return std::nullopt;
    return Replacement{
        index, {}, {1, in[1], out[2]}, {1, out[0], out[1], out[2]}};
  }
  // Preserve the old higher-benefit window-pool rule before other rank-4 rules.
  if (in[0] == 1 && in[1] > 1 && in[2] > 1 && in[3] > 1 &&
      in[1] <= 4096 / in[2] && in[1] * in[2] == out[0] && out[1] == in[3] &&
      out[2] == 1 && out[3] == 1 && out[0] <= 16777216 / out[1]) {
    auto n = out[0], c = out[1];
    std::vector<Shape> chain{{1, n, c}, {1, n, 1, c}, {1, n, c, 1}, {n, c, 1}};
    return Replacement{index, {}, chain.front(), {}, chain, 2};
  }
  if (in != out && in[0] == out[0]) {
    Shape ones, nonunit, permutation{0};
    for (int64_t i = 1; i < 4; ++i)
      (in[i] == 1 ? ones : nonunit).push_back(i);
    size_t unit = 0, other = 0;
    bool matches = true;
    for (int64_t i = 1; i < 4; ++i) {
      if (out[i] == 1) {
        if (unit >= ones.size()) {
          matches = false;
          break;
        }
        permutation.push_back(ones[unit++]);
      } else {
        if (other >= nonunit.size() || in[nonunit[other]] != out[i]) {
          matches = false;
          break;
        }
        permutation.push_back(nonunit[other++]);
      }
    }
    if (matches && unit == ones.size() && other == nonunit.size())
      return Replacement{index, std::move(permutation), {}};
  }
  for (auto d : in)
    if (d > 2147483647)
      return std::nullopt;
  Shape middle;
  if (in[0] == 1 && out[0] > 1 && in[1] / out[0] == out[1] &&
      in[1] % out[0] == 0 && in[2] == out[2] && in[3] == out[3])
    middle = {in[1], in[2], in[3]};
  else {
    if (in[0] != 1 || out[0] != 1 || out[1] <= 1 || out[2] <= 1 || out[3] <= 1)
      return std::nullopt;
    middle = {1};
    for (int i = 1; i < 4; ++i)
      if (in[i] != 1)
        middle.push_back(in[i]);
    if (middle.size() != 3)
      return std::nullopt;
    bool first = middle[1] / out[1] == out[2] && middle[1] % out[1] == 0 &&
                 middle[2] == out[3];
    bool second = middle[1] == out[1] && middle[2] / out[2] == out[3] &&
                  middle[2] % out[2] == 0;
    if (!first && !second)
      return std::nullopt;
  }
  return Replacement{index, {}, std::move(middle)};
}
class LowerLayout final : public mlir::RewritePattern {
  SdkSsaGraph &graph;
  Stats &stats;
  size_t index;
  const llvm::json::Array &plan;

public:
  LowerLayout(SdkSsaGraph &g, Stats &s, size_t i, const llvm::json::Array &p)
      : RewritePattern("vbx_sdk.node", 1, &g.context), graph(g), stats(s),
        index(i), plan(p) {
    setDebugName("LowerSdkReshapeLayoutSSA");
  }
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rw) const override {
    auto found = graph.original.find(op);
    if (found == graph.original.end() || found->second != index)
      return mlir::failure();
    ++stats.attempts;
    rw.setInsertionPoint(op);
    mlir::Value result;
    for (const auto &value : plan) {
      auto record = *value.getAsObject();
      llvm::SmallVector<mlir::Value> inputs;
      for (const auto &entry : *record.getArray("inputs")) {
        auto id = entry.getAsInteger().value();
        auto known = graph.values.find(id);
        if (known != graph.values.end())
          inputs.push_back(known->second);
        else {
          auto buffer = graph.tensor(id).getInteger("buffer").value();
          auto *data = (*graph.buffers)[buffer].getAsObject()->getArray("data");
          if (!data || data->empty())
            throw std::runtime_error("Missing layout SSA producer");
          inputs.push_back(graph.addConstant(rw, id));
        }
      }
      auto output = (*record.getArray("outputs"))[0].getAsInteger().value();
      result =
          graph.addNode(rw, std::move(record), inputs, output)->getResult(0);
    }
    graph.forgetNode(op);
    rw.replaceOp(op, result);
    ++stats.rewrites;
    return mlir::success();
  }
};
class LayoutSsaPass final
    : public mlir::PassWrapper<LayoutSsaPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  SdkSsaGraph &graph;
  Stats &stats;
  const std::map<size_t, llvm::json::Array> &plans;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutSsaPass)
  LayoutSsaPass(SdkSsaGraph &g, Stats &s,
                const std::map<size_t, llvm::json::Array> &p)
      : graph(g), stats(s), plans(p) {}
  void runOnOperation() override {
    std::map<size_t, mlir::Operation *> nodes;
    for (auto &entry : graph.original)
      nodes[entry.second] = entry.first;
    for (auto &[index, plan] : plans) {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<LowerLayout>(graph, stats, index, plan);
      mlir::FrozenRewritePatternSet frozen(std::move(patterns));
      mlir::GreedyRewriteConfig config;
      config.fold = false;
      config.cseConstants = false;
      config.maxNumRewrites = 2;
      llvm::SmallVector<mlir::Operation *> roots{nodes.at(index)};
      if (mlir::failed(mlir::applyOpPatternsGreedily(roots, frozen, config))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
Stats rewriteSdkLayoutJson(std::string &text, const Options &options,
                           std::string *dump) {
  if (options.max_rewrites < 1)
    throw std::runtime_error("max_rewrites must be positive");
  auto parsed = llvm::json::parse(text);
  if (!parsed)
    throw std::runtime_error("Invalid SDK JSON: " +
                             llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  if (!root)
    throw std::runtime_error("SDK JSON must be an object");
  auto *subgraphs = root->getArray("subgraphs");
  auto *codes = root->getArray("operator_codes");
  auto *buffers = root->getArray("buffers");
  if (!subgraphs || subgraphs->size() != 1 || !codes || !buffers)
    throw std::runtime_error(
        "SDK layout pass requires one subgraph and operator/buffer tables");
  auto *graph = (*subgraphs)[0].getAsObject();
  if (!graph)
    throw std::runtime_error("Invalid SDK subgraph");
  auto *ops = graph->getArray("operators");
  auto *tensors = graph->getArray("tensors");
  if (!ops || !tensors)
    throw std::runtime_error("Missing SDK operators/tensors");
  Stats stats;
  stats.nodes = ops->size();
  std::vector<Replacement> replacements;
  auto tensor = [&](int64_t i) -> llvm::json::Object * {
    return i >= 0 && size_t(i) < tensors->size() ? (*tensors)[i].getAsObject()
                                                 : nullptr;
  };
  // Graph IO and producer definitions do not change during candidate matching.
  // Index once instead of scanning every operator for every reshape.
  llvm::DenseSet<int64_t> runtimeShapeIds, mutatedIds;
  Shape graphInputs;
  if (integers(graph->getArray("inputs"), graphInputs))
    runtimeShapeIds.insert(graphInputs.begin(), graphInputs.end());
  // A signature input can replace a tensor's serialized initial value even
  // when it is not listed in the ordinary subgraph input table.
  if (auto *signatures = root->getArray("signature_defs"))
    for (auto &entry : *signatures) {
      auto *signature = entry.getAsObject();
      if (!signature)
        throw std::runtime_error("Invalid SDK signature");
      if (auto *inputs = signature->getArray("inputs"))
        for (auto &value : *inputs) {
          auto *input = value.getAsObject();
          auto id = input ? input->getInteger("tensor_index") : std::nullopt;
          if (!id || !tensor(*id))
            throw std::runtime_error("Invalid SDK signature input");
          runtimeShapeIds.insert(*id);
        }
    }
  for (auto &other : *ops)
    if (auto *object = other.getAsObject()) {
      if (auto *flags = object->getArray("mutating_variable_inputs")) {
        Shape inputs;
        if (!integers(object->getArray("inputs"), inputs) ||
            (!flags->empty() && flags->size() != inputs.size()))
          throw std::runtime_error("Invalid SDK mutation flags");
        for (size_t j = 0; j < flags->size(); ++j)
          if ((*flags)[j].getAsBoolean().value_or(true))
            mutatedIds.insert(inputs[j]);
      }
      if (auto *scratch = object->getArray("intermediates")) {
        Shape ids;
        if (!integers(scratch, ids))
          throw std::runtime_error("Invalid SDK intermediates");
        mutatedIds.insert(ids.begin(), ids.end());
      }
      Shape results;
      if (integers(object->getArray("outputs"), results))
        runtimeShapeIds.insert(results.begin(), results.end());
    }
  for (size_t i = 0; i < ops->size(); ++i) {
    auto *op = (*ops)[i].getAsObject();
    if (!op)
      continue;
    auto code = op->getInteger("opcode_index");
    if (!code || *code < 0 || size_t(*code) >= codes->size())
      continue;
    auto *desc = (*codes)[*code].getAsObject();
    if (!desc || desc->getString("builtin_code") != "RESHAPE")
      continue;
    Shape inputs, outputs, ishape, oshape;
    if (!integers(op->getArray("inputs"), inputs) ||
        !integers(op->getArray("outputs"), outputs) || inputs.size() != 2 ||
        outputs.size() != 1)
      continue;
    if (mutatedIds.contains(inputs[0]) || mutatedIds.contains(inputs[1]) ||
        mutatedIds.contains(outputs[0]))
      continue;
    if (auto *custom = op->getArray("custom_options"))
      if (!custom->empty())
        continue;
    if (auto *scratch = op->getArray("intermediates"))
      if (!scratch->empty())
        continue;
    if (op->getString("builtin_options_2_type").value_or("NONE") != "NONE" ||
        op->getInteger("large_custom_options_offset").value_or(0) ||
        op->getInteger("large_custom_options_size").value_or(0))
      continue;
    auto *in = tensor(inputs[0]);
    auto *out = tensor(outputs[0]);
    auto *target = tensor(inputs[1]);
    if (!in || !out || !target || !integers(in->getArray("shape"), ishape) ||
        !integers(out->getArray("shape"), oshape) || ishape.size() != 4 ||
        (oshape.size() != 4 && oshape.size() != 3))
      continue;
    if (in->getBoolean("is_variable").value_or(false) ||
        out->getBoolean("is_variable").value_or(false))
      continue;
    bool positive = true;
    for (auto d : ishape)
      positive &= d > 0;
    for (auto d : oshape)
      positive &= d > 0;
    if (!positive)
      continue;
    auto type = in->getString("type");
    if ((type != "INT8" && type != "UINT8") || type != out->getString("type"))
      continue;
    auto *iq = in->get("quantization");
    auto *oq = out->get("quantization");
    if (!iq || !oq || *iq != *oq)
      continue;
    auto *quant = iq->getAsObject();
    if (!quant)
      continue;
    auto *scales = quant->getArray("scale");
    if (!scales || scales->size() != 1)
      continue;
    Shape signature;
    bool dynamic = false;
    for (auto *t : {in, out})
      if (auto *sig = t->getArray("shape_signature")) {
        signature.clear();
        if (!integers(sig, signature) ||
            (!signature.empty() && signature != (t == in ? ishape : oshape)))
          dynamic = true;
      }
    if (dynamic || target->getString("type") != "INT32")
      continue;
    auto b = target->getInteger("buffer");
    if (!b || *b < 0 || size_t(*b) >= buffers->size())
      continue;
    auto *buffer = (*buffers)[*b].getAsObject();
    auto *data = buffer ? buffer->getArray("data") : nullptr;
    if (!data || data->size() != oshape.size() * 4 ||
        buffer->getInteger("offset").value_or(0) ||
        buffer->getInteger("size").value_or(0))
      continue;
    bool constant = true;
    for (size_t d = 0; d < oshape.size(); ++d) {
      uint32_t value = 0;
      for (size_t j = 0; j < 4; ++j) {
        auto byte = (*data)[4 * d + j].getAsInteger();
        if (!byte || *byte < 0 || *byte > 255) {
          constant = false;
          break;
        }
        value |= uint32_t(*byte) << (8 * j);
      }
      constant &= int64_t(value) == oshape[d];
    }
    if (!constant || target->getBoolean("is_variable").value_or(false))
      continue;
    if (runtimeShapeIds.contains(inputs[1]))
      continue;
    ++stats.candidates;
    if (auto plan = layoutPlan(i, ishape, oshape))
      replacements.push_back(std::move(*plan));
  }
  if (replacements.empty())
    return stats;
  if (replacements.size() > size_t(options.max_rewrites))
    throw std::runtime_error("Layout rewrite limit exceeded");
  SdkSsaGraph ssa(*root);
  std::map<size_t, llvm::json::Array> plans;
  // Match old greedy visitation for permutation metadata allocation, then
  // allocate multi-step chains in source order as before.
  std::reverse(replacements.begin(), replacements.end());
  size_t transposeCode = codes->size();
  for (size_t i = 0; i < codes->size(); ++i) {
    auto *c = (*codes)[i].getAsObject();
    if (c && c->getString("builtin_code") == "TRANSPOSE" &&
        c->getInteger("version").value_or(1) >= 2) {
      transposeCode = i;
      break;
    }
  }
  if (transposeCode == codes->size())
    codes->push_back(llvm::json::Object{{"builtin_code", "TRANSPOSE"},
                                        {"deprecated_builtin_code", 39},
                                        {"version", 2}});
  for (auto &replacement : replacements) {
    if (!replacement.intermediate.empty())
      continue;
    auto *op = (*ops)[replacement.index].getAsObject();
    auto input = (*op->getArray("inputs"))[0].getAsInteger().value();
    llvm::json::Array bytes;
    for (auto p : replacement.permutation)
      for (int j = 0; j < 4; ++j)
        bytes.push_back((p >> (8 * j)) & 255);
    auto buffer = buffers->size();
    buffers->push_back(llvm::json::Object{{"data", std::move(bytes)}});
    auto perm = tensors->size();
    tensors->push_back(llvm::json::Object{
        {"shape", array({4})},
        {"type", "INT32"},
        {"buffer", int64_t(buffer)},
        {"name", "__vbx_layout_perm_" + std::to_string(replacement.index)},
        {"is_variable", false},
        {"has_rank", true}});
    auto record = *op;
    record["opcode_index"] = int64_t(transposeCode);
    record["inputs"] = array({input, int64_t(perm)});
    record["builtin_options_type"] = "TransposeOptions";
    record["builtin_options"] = llvm::json::Object{};
    plans[replacement.index].push_back(std::move(record));
  }
  // Insert rank-changing steps only after all original operator indices have
  // been consumed; multiple replacements must not shift later source IDs.
  std::sort(replacements.begin(), replacements.end(),
            [](auto &a, auto &b) { return a.index < b.index; });
  size_t next = 0;
  for (size_t i = 0; i < ops->size(); ++i) {
    while (next < replacements.size() && replacements[next].index < i)
      ++next;
    if (next < replacements.size() && replacements[next].index == i &&
        !replacements[next].intermediate.empty()) {
      auto *op = (*ops)[i].getAsObject();
      auto input = (*op->getArray("inputs"))[0].getAsInteger().value();
      auto output = (*op->getArray("outputs"))[0].getAsInteger().value();
      std::vector<Shape> steps{replacements[next].intermediate};
      if (!replacements[next].extra.empty())
        steps.push_back(replacements[next].extra);
      if (!replacements[next].chain.empty())
        steps = replacements[next].chain;
      for (size_t step = 0; step < steps.size(); ++step) {
        auto &middle = steps[step];
        auto suffix = std::to_string(i) + "_" + std::to_string(step);
        llvm::json::Object temporary = *tensor(output);
        temporary["shape"] = array(middle);
        temporary["buffer"] = 0;
        temporary["name"] = "__vbx_layout_intermediate_" + suffix;
        if (temporary.getArray("shape_signature"))
          temporary["shape_signature"] = array(middle);
        auto temporaryId = tensors->size();
        tensors->push_back(std::move(temporary));
        auto parameter = int(step) == replacements[next].transposeStep
                             ? Shape{0, 1, 3, 2}
                             : middle;
        llvm::json::Array bytes;
        for (auto d : parameter)
          for (int j = 0; j < 4; ++j)
            bytes.push_back((d >> (8 * j)) & 255);
        auto buffer = buffers->size();
        buffers->push_back(llvm::json::Object{{"data", std::move(bytes)}});
        auto shapeId = tensors->size();
        tensors->push_back(
            llvm::json::Object{{"shape", array({int64_t(parameter.size())})},
                               {"type", "INT32"},
                               {"buffer", int64_t(buffer)},
                               {"name", "__vbx_layout_shape_" + suffix},
                               {"is_variable", false},
                               {"has_rank", true}});
        llvm::json::Object reshape = *op;
        reshape["inputs"] = array({input, int64_t(shapeId)});
        reshape["outputs"] = array({int64_t(temporaryId)});
        reshape["builtin_options_type"] = "ReshapeOptions";
        reshape["builtin_options"] =
            llvm::json::Object{{"new_shape", array(middle)}};
        if (int(step) == replacements[next].transposeStep) {
          reshape["opcode_index"] = int64_t(transposeCode);
          reshape["builtin_options_type"] = "TransposeOptions";
          reshape["builtin_options"] = llvm::json::Object{};
        }
        plans[i].push_back(std::move(reshape));
        input = temporaryId;
      }
      auto record = *op;
      record["inputs"] =
          array({input, (*op->getArray("inputs"))[1].getAsInteger().value()});
      plans[i].push_back(std::move(record));
    }
  }
  mlir::PassManager pm(&ssa.context);
  pm.enableVerifier(true);
  pm.addPass(std::make_unique<LayoutSsaPass>(ssa, stats, plans));
  if (mlir::failed(pm.run(*ssa.module)) ||
      stats.rewrites != replacements.size())
    throw std::runtime_error("Layout SSA rewrite failed");
  if (dump) {
    llvm::raw_string_ostream stream(*dump);
    ssa.module->print(stream);
  }
  ssa.exportOperators();
  text.clear();
  llvm::raw_string_ostream stream(text);
  llvm::json::OStream json(stream, 2);
  writeSdkValue(*parsed, json);
  return stats;
}
} // namespace vbx
