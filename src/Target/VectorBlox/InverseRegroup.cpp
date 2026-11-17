#include "Graph.h"
#include "vbx_rewrite/engine.h"
#include <cmath>
namespace vbx {
namespace {
using Ids = std::vector<int64_t>;
struct NoMatch {};
void need(bool b) {
  if (!b)
    throw NoMatch{};
}
Ids ints(const llvm::json::Array *a) {
  need(a);
  Ids r;
  for (auto &v : *a) {
    auto n = v.getAsInteger();
    need(bool(n));
    r.push_back(*n);
  }
  return r;
}
llvm::json::Array arr(const Ids &v) {
  llvm::json::Array a;
  for (auto n : v)
    a.push_back(n);
  return a;
}
class LowerInverseRegroup final : public mlir::RewritePattern {
  SdkSsaGraph &g;
  Stats &stats;
  const Options &options;
  int64_t id(mlir::Value v) const {
    need(g.tensorIds.count(v));
    return g.tensorIds.lookup(v);
  }
  llvm::json::Object &buffer(int64_t t) const {
    auto b = g.tensor(t).getInteger("buffer");
    need(b && *b >= 0 && size_t(*b) < g.buffers->size());
    auto *o = (*g.buffers)[*b].getAsObject();
    need(o && !o->getInteger("offset").value_or(0) &&
         !o->getInteger("size").value_or(0));
    return *o;
  }
  Ids shape(int64_t i) const {
    auto &t = g.tensor(i);
    need(!g.mutated.count(i) && !t.getBoolean("is_variable").value_or(false));
    auto s = ints(t.getArray("shape"));
    if (auto *sig = t.getArray("shape_signature"))
      need(ints(sig) == s);
    for (auto d : s)
      need(d > 0 && d <= INT32_MAX);
    return s;
  }
  void runtime(int64_t i) const {
    auto &t = g.tensor(i);
    need(t.getString("type") == "INT8");
    auto *data = buffer(i).getArray("data");
    need(!data || data->empty());
    auto *q = t.getObject("quantization");
    need(q && q->getString("details_type").value_or("NONE") == "NONE");
    auto *scale = q->getArray("scale");
    auto *zero = q->getArray("zero_point");
    need(scale && scale->size() == 1 && zero && zero->size() == 1);
    auto s = (*scale)[0].getAsNumber();
    auto z = (*zero)[0].getAsInteger();
    need(s && std::isnormal(float(*s)) && *s > 0 && z && *z >= -128 &&
         *z <= 127);
  }
  void plain(mlir::Operation *op, llvm::StringRef name, int64_t version) const {
    need(op && op->getName().getStringRef() == "vbx_sdk.node");
    auto &rec = g.record(op);
    auto *code =
        (*g.codes)[rec.getInteger("opcode_index").value()].getAsObject();
    need(code && code->getString("builtin_code") == name &&
         code->getInteger("version").value_or(1) == version);
    for (auto key : {"custom_options", "intermediates"})
      if (auto *a = rec.getArray(key))
        need(a->empty());
    if (auto *a = rec.getArray("mutating_variable_inputs"))
      for (auto &v : *a)
        need(!v.getAsBoolean().value_or(true));
    need(rec.getString("builtin_options_2_type").value_or("NONE") == "NONE" &&
         !rec.getInteger("large_custom_options_offset").value_or(0) &&
         !rec.getInteger("large_custom_options_size").value_or(0));
  }

public:
  LowerInverseRegroup(mlir::MLIRContext *c, SdkSsaGraph &graph, Stats &s,
                      const Options &o)
      : RewritePattern("vbx_sdk.node", 1, c), g(graph), stats(s), options(o) {
    setDebugName("LowerProducedInverseRegroup");
  }
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rw) const override {
    auto kind = op->getAttrOfType<mlir::StringAttr>("builtin");
    if (!kind || kind.getValue() != "RESHAPE")
      return mlir::failure();
    ++stats.attempts;
    try {
      plain(op, "RESHAPE", 1);
      need(op->getNumOperands() == 2 && op->getNumResults() == 1);
      auto input = op->getOperand(0);
      auto src = id(input), dst = id(op->getResult(0)),
           dims = id(op->getOperand(1));
      auto x = shape(src), y = shape(dst);
      need(x.size() == 3 && y.size() == 4 && x[0] == 1 && y[0] == 1 &&
           x[1] <= 4096 && x[2] <= 8192 && x[2] <= options.max_channels &&
           y[1] <= 4096 && y[2] <= 4096 && y[3] >= 2 && y[3] <= 32 &&
           y[3] < x[2] && x[2] % y[3] == 0);
      auto groups = x[2] / y[3];
      need(groups <= 256 && x[1] * x[2] <= 1048576 &&
           x[1] * x[2] == y[1] * y[2] * y[3]);
      // Root-input SLICE fanout followed by CONCAT is not numerically reliable
      // on SDK 3.1. Keep the verified, private materialized TRANSPOSE producer.
      need(input.hasOneUse() && !g.exposed.count(src));
      auto *producer = input.getDefiningOp();
      need(producer && producer->getName().getStringRef() == "vbx_sdk.node");
      auto *pc =
          (*g.codes)[g.record(producer).getInteger("opcode_index").value()]
              .getAsObject();
      auto pv = pc->getInteger("version").value_or(1);
      need(pv == 1 || pv == 2 || pv == 4);
      plain(producer, "TRANSPOSE", pv);
      runtime(src);
      runtime(dst);
      need(*g.tensor(src).get("quantization") ==
           *g.tensor(dst).get("quantization"));
      need(shape(dims) == Ids{4} &&
           g.tensor(dims).getString("type") == "INT32" &&
           !g.exposed.count(dims));
      auto *def = op->getOperand(1).getDefiningOp();
      need(def && def->getName().getStringRef() == "vbx_sdk.source");
      auto bytes = ints(buffer(dims).getArray("data"));
      need(bytes.size() == 16);
      Ids decoded;
      for (unsigned i = 0; i < 16; i += 4) {
        uint32_t n = 0;
        for (unsigned j = 0; j < 4; ++j) {
          need(bytes[i + j] >= 0 && bytes[i + j] <= 255);
          n |= uint32_t(bytes[i + j]) << (8 * j);
        }
        decoded.push_back(int32_t(n));
      }
      need(decoded == y);
      if (auto *opts = g.record(op).getObject("builtin_options"))
        if (auto *shape = opts->getArray("new_shape"))
          need(shape->empty() || ints(shape) == y);
      ++stats.candidates;
      if (stats.rewrites >= size_t(options.max_rewrites))
        throw std::runtime_error("SDK SSA rewrite limit exceeded");
      auto prefix = "__vbx_inverse_regroup_" + std::to_string(dst) + "_";
      auto constant = [&](const Ids &values) {
        Ids bytes;
        for (auto n : values)
          for (unsigned j = 0; j < 4; ++j)
            bytes.push_back((uint32_t(n) >> (8 * j)) & 255);
        auto b = int64_t(g.buffers->size());
        g.buffers->push_back(llvm::json::Object{{"data", arr(bytes)}});
        auto t = int64_t(g.tensors->size());
        g.tensors->push_back(
            llvm::json::Object{{"name", prefix + std::to_string(t)},
                               {"shape", arr({int64_t(values.size())})},
                               {"type", "INT32"},
                               {"buffer", b}});
        return g.addConstant(rw, t);
      };
      auto tensor = [&](const Ids &shape) {
        auto rec = g.tensor(src);
        auto t = int64_t(g.tensors->size());
        auto b = int64_t(g.buffers->size());
        g.buffers->push_back(llvm::json::Object{});
        rec["name"] = prefix + std::to_string(t);
        rec["shape"] = arr(shape);
        if (rec.getArray("shape_signature"))
          rec["shape_signature"] = arr(shape);
        rec["buffer"] = b;
        g.tensors->push_back(std::move(rec));
        return t;
      };
      auto code = [&](llvm::StringRef name, int number, int version) {
        for (size_t i = 0; i < g.codes->size(); ++i) {
          auto *c = (*g.codes)[i].getAsObject();
          if (c && c->getString("builtin_code") == name &&
              c->getInteger("version").value_or(1) == version)
            return int64_t(i);
        }
        auto i = int64_t(g.codes->size());
        g.codes->push_back(
            llvm::json::Object{{"builtin_code", name},
                               {"deprecated_builtin_code", number},
                               {"version", version}});
        return i;
      };
      auto reshapeCode = code("RESHAPE", 22, 1),
           sliceCode = code("SLICE", 65, 2),
           concatCode = code("CONCATENATION", 2, 2);
      auto reshape = [&](mlir::Value input, const Ids &shape, int64_t output) {
        auto shapeValue = constant(shape);
        return g
            .addNode(rw,
                     llvm::json::Object{
                         {"opcode_index", reshapeCode},
                         {"builtin_options_type", "ReshapeOptions"},
                         {"builtin_options",
                          llvm::json::Object{{"new_shape", arr(shape)}}}},
                     mlir::ValueRange{input, shapeValue}, output)
            ->getResult(0);
      };
      llvm::SmallVector<mlir::Value> pieces;
      auto size = constant({1, x[1], y[3]});
      for (int64_t i = 0; i < groups; ++i) {
        auto begin = constant({0, 0, i * y[3]});
        auto t = tensor({1, x[1], y[3]});
        auto sliced = g.addNode(rw,
                                llvm::json::Object{
                                    {"opcode_index", sliceCode},
                                    {"builtin_options_type", "SliceOptions"},
                                    {"builtin_options", llvm::json::Object{}}},
                                mlir::ValueRange{input, begin, size}, t)
                          ->getResult(0);
        pieces.push_back(
            reshape(sliced, {1, x[1], 1, y[3]}, tensor({1, x[1], 1, y[3]})));
      }
      auto joined =
          g.addNode(
               rw,
               llvm::json::Object{
                   {"opcode_index", concatCode},
                   {"builtin_options_type", "ConcatenationOptions"},
                   {"builtin_options",
                    llvm::json::Object{{"axis", 2},
                                       {"fused_activation_function", "NONE"}}}},
               pieces, tensor({1, x[1], groups, y[3]}))
              ->getResult(0);
      auto flat = reshape(joined, {1, x[1] * groups, y[3]},
                          tensor({1, x[1] * groups, y[3]}));
      auto result = reshape(flat, y, dst);
      g.forgetNode(op);
      rw.replaceOp(op, result);
      ++stats.rewrites;
      return mlir::success();
    } catch (const NoMatch &) {
      return mlir::failure();
    }
  }
};
} // namespace
void populateSdkInverseRegroupPatterns(mlir::RewritePatternSet &patterns,
                                       SdkSsaGraph &graph, Stats &stats,
                                       const Options &options) {
  patterns.add<LowerInverseRegroup>(&graph.context, graph, stats, options);
}
} // namespace vbx
