#include "Graph.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "vbx_rewrite/engine.h"
#include <cmath>
#include <stdexcept>

namespace vbx {
namespace {
using Dims = std::vector<int64_t>;
struct NoMatch {};
void need(bool value) {
  if (!value)
    throw NoMatch{};
}
Dims ints(const llvm::json::Array *a) {
  need(a);
  Dims result;
  for (auto &v : *a) {
    auto n = v.getAsInteger();
    need(bool(n));
    result.push_back(*n);
  }
  return result;
}
// Only validated, static, quantization-preserving views are represented here.
// Generic SDK nodes remain conservative. This op has no memory side effects:
// its shape changes the tensor interpretation, never its linear element order.
class View final
    : public mlir::Op<View, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult,
                      mlir::OpTrait::ZeroRegions,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(View)
  static llvm::StringRef getOperationName() { return "vbx_view.reshape"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
  void
  getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &) {}
  mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute>) {
    auto previous = mlir::dyn_cast_or_null<View>(getOperand().getDefiningOp());
    if (previous &&
        previous.getOperand().getType() == (*this)->getResult(0).getType())
      return previous.getOperand();
    return {};
  }
};
class ViewDialect final : public mlir::Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ViewDialect)
  static llvm::StringRef getDialectNamespace() { return "vbx_view"; }
  explicit ViewDialect(mlir::MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                mlir::TypeID::get<ViewDialect>()) {
    addOperations<View>();
  }
};
class Eligibility {
  SdkSsaGraph &g;

public:
  explicit Eligibility(SdkSsaGraph &graph) : g(graph) {}
  int64_t id(mlir::Value value) {
    need(g.tensorIds.count(value));
    return g.tensorIds.lookup(value);
  }
  Dims shape(mlir::Value value) {
    auto i = id(value);
    auto &t = g.tensor(i);
    need(!g.mutated.count(i) && !t.getBoolean("is_variable").value_or(false));
    auto ds = ints(t.getArray("shape"));
    need(!ds.empty() && ds.size() <= 6);
    int64_t count = 1;
    for (auto d : ds) {
      need(d > 0 && d <= 16777216 / count);
      count *= d;
    }
    if (auto *a = t.getArray("shape_signature"))
      need(ints(a) == ds);
    return ds;
  }
  llvm::json::Object &buffer(mlir::Value value) {
    auto b = g.tensor(id(value)).getInteger("buffer");
    need(b && *b >= 0 && size_t(*b) < g.buffers->size());
    auto *r = (*g.buffers)[*b].getAsObject();
    need(r && !r->getInteger("offset").value_or(0) &&
         !r->getInteger("size").value_or(0));
    return *r;
  }
  void data(mlir::Value value, mlir::Value source) {
    shape(value);
    auto &t = g.tensor(id(value));
    auto &s = g.tensor(id(source));
    auto type = t.getString("type");
    need((type == "INT8" || type == "UINT8") && type == s.getString("type"));
    auto *q = t.getObject("quantization"), *sq = s.getObject("quantization");
    need(q && sq && *q == *sq &&
         q->getString("details_type").value_or("NONE") == "NONE");
    auto *scales = q->getArray("scale"), *zeros = q->getArray("zero_point");
    need(scales && zeros && scales->size() == 1 && zeros->size() == 1);
    auto scale = (*scales)[0].getAsNumber();
    auto zero = (*zeros)[0].getAsInteger();
    need(scale && *scale > 0 && std::isnormal(float(*scale)) && zero &&
         (type == "INT8" ? *zero >= -128 && *zero <= 127
                         : *zero >= 0 && *zero <= 255));
    auto *bytes = buffer(value).getArray("data");
    need(!bytes || bytes->empty());
  }
  void reshape(mlir::Operation *op) {
    need(op && op->getName().getStringRef() == "vbx_sdk.node" &&
         op->getNumOperands() == 2 && op->getNumResults() == 1);
    auto &r = g.record(op);
    auto ci = r.getInteger("opcode_index");
    need(ci && *ci >= 0 && size_t(*ci) < g.codes->size());
    auto *c = (*g.codes)[*ci].getAsObject();
    need(c && c->getString("builtin_code") == "RESHAPE" &&
         c->getInteger("version").value_or(1) == 1);
    for (auto key : {"custom_options", "intermediates"})
      if (auto *a = r.getArray(key))
        need(a->empty());
    if (auto *a = r.getArray("mutating_variable_inputs"))
      for (auto &v : *a)
        need(!v.getAsBoolean().value_or(true));
    need(r.getString("builtin_options_2_type").value_or("NONE") == "NONE" &&
         !r.getInteger("large_custom_options_offset").value_or(0) &&
         !r.getInteger("large_custom_options_size").value_or(0));
    auto out = shape(op->getResult(0));
    auto in = shape(op->getOperand(0));
    int64_t a = 1, b = 1;
    for (auto d : in)
      a *= d;
    for (auto d : out)
      b *= d;
    need(a == b);
    auto parameter = op->getOperand(1);
    auto *def = parameter.getDefiningOp();
    need(def && def->getName().getStringRef() == "vbx_sdk.source" &&
         !g.exposed.count(id(parameter)) &&
         g.tensor(id(parameter)).getString("type") == "INT32" &&
         shape(parameter) == Dims{int64_t(out.size())});
    auto bytes = ints(buffer(parameter).getArray("data"));
    need(bytes.size() == out.size() * 4);
    for (size_t i = 0; i < out.size(); ++i) {
      uint32_t n = 0;
      for (unsigned j = 0; j < 4; ++j) {
        auto v = bytes[4 * i + j];
        need(v >= 0 && v <= 255);
        n |= uint32_t(v) << (8 * j);
      }
      need(int64_t(int32_t(n)) == out[i]);
    }
    auto type = r.getString("builtin_options_type").value_or("NONE");
    need(type == "NONE" || type == "ReshapeOptions");
    if (auto *o = r.getObject("builtin_options"))
      if (auto *s = o->getArray("new_shape"))
        need(s->empty() || ints(s) == out);
  }
};
} // namespace

void canonicalizeSdkInverseViews(SdkSsaGraph &g, Stats &stats,
                                 const Options &options) {
  Eligibility check(g);
  std::vector<std::pair<mlir::Operation *, mlir::Operation *>> pairs;
  std::set<mlir::Operation *> claimed;
  for (auto &op : g.function.getBody().front())
    try {
      auto kind = op.getAttrOfType<mlir::StringAttr>("builtin");
      need(kind && kind.getValue() == "RESHAPE");
      check.reshape(&op);
      auto *first = op.getOperand(0).getDefiningOp();
      check.reshape(first);
      need(!claimed.count(first) && !claimed.count(&op));
      mlir::Value source = first->getOperand(0), middle = first->getResult(0),
                  end = op.getResult(0);
      need(middle == op.getOperand(0) && middle.hasOneUse() &&
           !g.exposed.count(check.id(middle)) &&
           !g.exposed.count(check.id(end)));
      need(check.shape(source) == check.shape(end) &&
           check.shape(source) != check.shape(middle));
      check.data(source, source);
      check.data(middle, source);
      check.data(end, source);
      pairs.emplace_back(first, &op);
      claimed.insert(first);
      claimed.insert(&op);
    } catch (const NoMatch &) {
    }
  if (pairs.empty())
    return;
  stats.candidates += pairs.size();
  if (stats.rewrites + pairs.size() > size_t(options.max_rewrites))
    throw std::runtime_error("Inverse-view rewrite limit exceeded");
  g.context.getOrLoadDialect<ViewDialect>();
  mlir::IRRewriter rw(&g.context);
  for (auto [first, last] : pairs) {
    rw.setInsertionPoint(first);
    mlir::OperationState a(first->getLoc(), View::getOperationName());
    a.addOperands(first->getOperand(0));
    a.addTypes(first->getResultTypes());
    auto *view = rw.create(a);
    rw.setInsertionPoint(last);
    mlir::OperationState b(last->getLoc(), View::getOperationName());
    b.addOperands(view->getResult(0));
    b.addTypes(last->getResultTypes());
    auto *inverse = rw.create(b);
    g.forgetNode(last);
    rw.replaceOp(last, inverse->getResults());
    g.forgetNode(first);
    rw.eraseOp(first);
  }
  // Upstream folding replaces the inverse view with its original SSA value;
  // standard dead-operation removal erases the now-unused first view.
  mlir::PassManager pm(&g.context);
  pm.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(*g.module)))
    throw std::runtime_error("Inverse-view canonicalization failed");
  for (auto &op : g.function.getBody().front())
    if (mlir::isa<View>(op))
      throw std::runtime_error("Unfolded SDK view cannot be exported");
  stats.rewrites += pairs.size();
}
} // namespace vbx
