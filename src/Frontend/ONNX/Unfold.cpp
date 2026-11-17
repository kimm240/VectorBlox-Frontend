#include "Bridge.h"
#include "Patterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <cstring>

namespace vbx {
namespace {
// Matches the static, non-overlapping single-channel PyTorch unfold export.
// Attribute/constant checks prove ordering; tensor names are not predicates.
class Unfold final : public mlir::RewritePattern {
  Bridge &b;
  const onnx::NodeProto *node(mlir::Operation *op, const char *kind) const {
    if (!op)
      return nullptr;
    auto id = op->getAttrOfType<mlir::IntegerAttr>("vbx.source");
    if (!id)
      return nullptr;
    auto &n = b.model.graph().node(id.getInt());
    return n.domain().empty() && n.op_type() == kind && n.output_size() == 1
               ? &n
               : nullptr;
  }
  bool ints(mlir::Value v, llvm::ArrayRef<int64_t> expected) const {
    auto *t = b.constants.lookup(v);
    if (!t || t->data_type() != onnx::TensorProto::INT64 ||
        t->dims_size() != 1 || t->dims(0) != int64_t(expected.size()) ||
        t->external_data_size() ||
        t->data_location() == onnx::TensorProto::EXTERNAL)
      return false;
    if (t->has_raw_data()) {
      if (t->raw_data().size() != expected.size() * 8)
        return false;
      for (size_t j = 0; j < expected.size(); ++j) {
        uint64_t bits = 0;
        for (unsigned k = 0; k < 8; ++k)
          bits |= uint64_t(static_cast<unsigned char>(t->raw_data()[j * 8 + k]))
                  << (8 * k);
        int64_t value;
        std::memcpy(&value, &bits, 8);
        if (value != expected[j])
          return false;
      }
      return true;
    }
    return size_t(t->int64_data_size()) == expected.size() &&
           std::equal(expected.begin(), expected.end(),
                      t->int64_data().begin());
  }
  bool attr(const onnx::NodeProto *n, const char *name,
            llvm::ArrayRef<int64_t> values, bool scalar = false) const {
    if (!n || n->attribute_size() != 1)
      return false;
    auto &a = n->attribute(0);
    if (a.name() != name)
      return false;
    if (scalar)
      return a.type() == onnx::AttributeProto::INT && a.i() == values[0];
    return a.type() == onnx::AttributeProto::INTS &&
           size_t(a.ints_size()) == values.size() &&
           std::equal(values.begin(), values.end(), a.ints().begin());
  }
  bool shape(mlir::Value value, llvm::ArrayRef<int64_t> dims) const {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    return t && t.getElementType().isF32() && t.getShape() == dims;
  }
  bool reshape(mlir::Operation *op, llvm::ArrayRef<int64_t> dims,
               bool inferChannel = false) const {
    auto *n = node(op, "Reshape");
    if (!n || op->getNumOperands() != 2)
      return false;
    if (n->attribute_size() && !attr(n, "allowzero", {0}, true))
      return false;
    if (ints(op->getOperand(1), dims))
      return true;
    auto inferred = llvm::SmallVector<int64_t>(dims);
    if (inferChannel)
      inferred[1] = -1;
    return inferChannel && ints(op->getOperand(1), inferred);
  }
  // Match Slice -> Transpose -> Unsqueeze branches concatenated in order.
  mlir::Value
  unfoldAxis(mlir::Operation *concat, int64_t axis, int64_t count,
             int64_t block, llvm::ArrayRef<int64_t> inputShape,
             llvm::SmallPtrSetImpl<mlir::Operation *> &matched) const {
    if (!attr(node(concat, "Concat"), "axis", {axis}, true) ||
        int64_t(concat->getNumOperands()) != count)
      return {};
    mlir::Value input;
    llvm::SmallVector<int64_t> perm;
    for (int64_t i = 0; i < int64_t(inputShape.size()); ++i)
      if (i != axis)
        perm.push_back(i);
    perm.push_back(axis);
    for (int64_t j = 0; j < count; ++j) {
      auto *u = concat->getOperand(j).getDefiningOp();
      auto *un = node(u, "Unsqueeze");
      if (!un || un->attribute_size() || u->getNumOperands() != 2 ||
          !ints(u->getOperand(1), {axis}))
        return {};
      auto *t = u->getOperand(0).getDefiningOp();
      if (!attr(node(t, "Transpose"), "perm", perm) || t->getNumOperands() != 1)
        return {};
      auto *s = t->getOperand(0).getDefiningOp();
      auto *sn = node(s, "Slice");
      if (!sn || sn->attribute_size() ||
          (s->getNumOperands() != 4 && s->getNumOperands() != 5))
        return {};
      if (!ints(s->getOperand(1), {j * block}) ||
          !ints(s->getOperand(2), {(j + 1) * block}) ||
          !ints(s->getOperand(3), {axis}) ||
          (s->getNumOperands() == 5 && !ints(s->getOperand(4), {1})))
        return {};
      if (j == 0)
        input = s->getOperand(0);
      if (input != s->getOperand(0) || !shape(input, inputShape))
        return {};
      auto sliced = llvm::SmallVector<int64_t>(inputShape);
      sliced[axis] = block;
      llvm::SmallVector<int64_t> transposed;
      for (auto p : perm)
        transposed.push_back(sliced[p]);
      auto unsqueezed = transposed;
      unsqueezed.insert(unsqueezed.begin() + axis, 1);
      if (!shape(s->getResult(0), sliced) ||
          !shape(t->getResult(0), transposed) ||
          !shape(u->getResult(0), unsqueezed))
        return {};
      matched.insert(s);
      matched.insert(t);
      matched.insert(u);
    }
    matched.insert(concat);
    return input;
  }

public:
  Unfold(mlir::MLIRContext *ctx, Bridge &bridge)
      : RewritePattern("vbx_onnx.Reshape", 2, ctx), b(bridge) {
    setDebugName("StaticUnfoldToConv");
  }
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *root,
                  mlir::PatternRewriter &rw) const override {
    ++b.stats.attempts;
    auto reject = [&]() {
      ++b.stats.skipped["not_supported_static_unfold"];
      return mlir::failure();
    };
    if (b.opset < 13 || b.opset > 21 || root->getNumOperands() != 2 ||
        root->getNumResults() != 1)
      return reject();
    // Nested captures are opaque to this adapter: do not remove intermediate
    // nodes.
    for (auto &n : b.model.graph().node())
      for (auto &a : n.attribute())
        if (a.type() == onnx::AttributeProto::GRAPH ||
            a.type() == onnx::AttributeProto::GRAPHS)
          return reject();
    auto *trans = root->getOperand(0).getDefiningOp();
    if (!attr(node(trans, "Transpose"), "perm", {0, 1, 4, 2, 3}) ||
        trans->getNumOperands() != 1)
      return reject();
    auto *first = trans->getOperand(0).getDefiningOp();
    if (!node(first, "Reshape") || first->getNumOperands() != 2)
      return reject();
    auto source = first->getOperand(0);
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
    if (!type || type.getRank() != 6 || !type.hasStaticShape() ||
        !type.getElementType().isF32())
      return reject();
    auto d = type.getShape();
    int64_t batch = d[0], h = d[2], w = d[3], k = d[4];
    if (batch != 1 || d[1] != 1 || h < 1 || w < 1 || h > 4096 || w > 4096 ||
        k < 2 || k > 8 || d[5] != k)
      return reject();
    if (!reshape(first, {batch, 1, h, w, k * k}) ||
        !shape(first->getResult(0), {batch, 1, h, w, k * k}) ||
        !shape(trans->getResult(0), {batch, 1, k * k, h, w}) ||
        !reshape(root, {batch, k * k, h, w}, true) ||
        !shape(root->getResult(0), {batch, k * k, h, w}))
      return reject();
    llvm::SmallPtrSet<mlir::Operation *, 32> matched;
    auto columns = unfoldAxis(source.getDefiningOp(), 3, w, k,
                              {batch, 1, h, w * k, k}, matched);
    if (!columns)
      return reject();
    auto input = unfoldAxis(columns.getDefiningOp(), 2, h, k,
                            {batch, 1, h * k, w * k}, matched);
    if (!input)
      return reject();
    matched.insert(first);
    matched.insert(trans);
    matched.insert(root);
    for (auto *op : matched)
      if (op != root)
        for (auto *user : op->getUsers())
          if (!matched.contains(user))
            return reject();
    int64_t id = root->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt();
    auto *firstSlice = columns.getDefiningOp()
                           ->getOperand(0)
                           .getDefiningOp()
                           ->getOperand(0)
                           .getDefiningOp()
                           ->getOperand(0)
                           .getDefiningOp();
    std::string inputName = node(firstSlice, "Slice")->input(0);
    auto name = b.uniqueWeightName(id);
    mlir::OperationState ws(root->getLoc(), "vbx_onnx.generated_weight");
    ws.addTypes(mlir::RankedTensorType::get({k * k, 1, k, k}, rw.getF32Type()));
    ws.addAttribute("vbx.name", rw.getStringAttr(name));
    auto *weight = rw.create(ws);
    mlir::OperationState cs(root->getLoc(), "vbx_onnx.Conv");
    cs.addOperands({input, weight->getResult(0)});
    cs.addTypes(root->getResultTypes());
    cs.addAttribute("vbx.source", rw.getI64IntegerAttr(id));
    cs.addAttribute("vbx.generated_weight", rw.getStringAttr(name));
    cs.addAttribute("vbx.input", rw.getStringAttr(inputName));
    cs.addAttribute("vbx.block", rw.getI64IntegerAttr(k));
    auto *conv = rw.create(cs);
    std::vector<mlir::Operation *> erased;
    for (auto *op : matched)
      if (op != root)
        erased.push_back(op);
    std::sort(erased.begin(), erased.end(), [](auto *a, auto *c) {
      return a->template getAttrOfType<mlir::IntegerAttr>("vbx.source")
                 .getInt() >
             c->template getAttrOfType<mlir::IntegerAttr>("vbx.source")
                 .getInt();
    });
    rw.replaceOp(root, conv->getResults());
    for (auto *op : erased) {
      b.removed.insert(
          op->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt());
      rw.eraseOp(op);
    }
    b.generated.push_back({name, 1, k});
    ++b.stats.rewrites;
    return mlir::success();
  }
};
} // namespace
void populateUnfoldPatterns(mlir::RewritePatternSet &patterns, Bridge &bridge) {
  patterns.add<Unfold>(patterns.getContext(), bridge);
}
} // namespace vbx
