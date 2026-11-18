#include "Bridge.h"
#include "Patterns.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace vbx {
namespace {
using Shape = std::vector<int64_t>;
const onnx::NodeProto *source(Bridge &b, mlir::Operation *op,
                              const char *kind) {
  if (!op)
    return nullptr;
  auto id = op->getAttrOfType<mlir::IntegerAttr>("vbx.source");
  if (!id)
    return nullptr;
  const auto &n = b.model.graph().node(id.getInt());
  return n.domain().empty() && n.op_type() == kind ? &n : nullptr;
}
bool readFloats(const onnx::TensorProto *t, Shape &dims,
                std::vector<float> &v) {
  if (!t || t->data_type() != onnx::TensorProto::FLOAT ||
      t->data_location() == onnx::TensorProto::EXTERNAL ||
      t->external_data_size())
    return false;
  int64_t count = 1;
  for (auto d : t->dims()) {
    if (d < 1 || count > 16777216 / d)
      return false;
    count *= d;
    dims.push_back(d);
  }
  v.resize(count);
  if (t->has_raw_data()) {
    if (t->raw_data().size() != size_t(count * 4))
      return false;
    for (int64_t i = 0; i < count; ++i) {
      uint32_t bits = 0;
      for (unsigned j = 0; j < 4; ++j)
        bits |= uint32_t(static_cast<unsigned char>(t->raw_data()[4 * i + j]))
                << (8 * j);
      std::memcpy(&v[i], &bits, 4);
    }
  } else {
    if (t->float_data_size() != count)
      return false;
    std::copy(t->float_data().begin(), t->float_data().end(), v.begin());
  }
  return std::all_of(v.begin(), v.end(),
                     [](float x) { return std::isfinite(x); });
}
struct Branch {
  mlir::Operation *op;
  const onnx::NodeProto *node;
  Shape kernel, stride{1, 1}, pads{0, 0, 0, 0}, dims;
  std::vector<float> weights, bias;
};
bool readBranch(Bridge &b, mlir::Value value, Branch &branch) {
  branch.op = value.getDefiningOp();
  branch.node = source(b, branch.op, "Conv");
  if (!branch.node || !value.hasOneUse() || branch.op->getNumResults() != 1 ||
      branch.node->input_size() < 2 || branch.node->input_size() > 3)
    return false;
  if (!readFloats(b.constants.lookup(branch.op->getOperand(1)), branch.dims,
                  branch.weights) ||
      branch.dims.size() != 4)
    return false;
  branch.kernel = {branch.dims[2], branch.dims[3]};
  for (auto k : branch.kernel)
    if (k < 1 || k > 7 || k % 2 != 1)
      return false;
  std::set<std::string> seen;
  for (const auto &a : branch.node->attribute()) {
    if (!seen.insert(a.name()).second)
      return false;
    if (a.name() == "group") {
      if (a.type() != onnx::AttributeProto::INT || a.i() != 1)
        return false;
    } else if (a.name() == "auto_pad") {
      if (a.type() != onnx::AttributeProto::STRING || a.s() != "NOTSET")
        return false;
    } else {
      if (a.type() != onnx::AttributeProto::INTS)
        return false;
      Shape data(a.ints().begin(), a.ints().end());
      if (a.name() == "kernel_shape") {
        if (data != branch.kernel)
          return false;
      } else if (a.name() == "strides")
        branch.stride = data;
      else if (a.name() == "pads")
        branch.pads = data;
      else if (a.name() == "dilations") {
        if (data != Shape{1, 1})
          return false;
      } else
        return false;
    }
  }
  if (branch.stride.size() != 2 || branch.pads.size() != 4)
    return false;
  for (auto s : branch.stride)
    if (s < 1 || s > 16)
      return false;
  for (unsigned i = 0; i < 4; ++i)
    if (branch.pads[i] != branch.kernel[i % 2] / 2)
      return false;
  branch.bias.assign(branch.dims[0], 0.f);
  if (branch.node->input_size() == 3 && !branch.node->input(2).empty()) {
    Shape dims;
    if (!readFloats(b.constants.lookup(branch.op->getOperand(2)), dims,
                    branch.bias) ||
        dims != Shape{branch.dims[0]})
      return false;
  }
  return true;
}
class ConvBranches final : public mlir::RewritePattern {
  Bridge &b;

public:
  ConvBranches(mlir::MLIRContext *context, Bridge &bridge)
      : RewritePattern("vbx_onnx.Add", 2, context), b(bridge) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *root,
                  mlir::PatternRewriter &rw) const override {
    ++b.stats.attempts;
    const auto *add = source(b, root, "Add");
    if (b.opset < 13 || b.opset > 21 || !add || root->getNumOperands() != 2 ||
        root->getNumResults() != 1 || add->attribute_size())
      return mlir::failure();
    auto id = root->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt();
    if (b.orderSensitive.count(id))
      return mlir::failure();
    // Nested graphs may capture branch outputs without an SSA use here.
    for (const auto &n : b.model.graph().node())
      for (const auto &attr : n.attribute())
        if (attr.type() == onnx::AttributeProto::GRAPH ||
            attr.type() == onnx::AttributeProto::GRAPHS)
          return mlir::failure();
    Branch a, c;
    if (!readBranch(b, root->getOperand(0), a) ||
        !readBranch(b, root->getOperand(1), c) || a.op == c.op ||
        a.op->getOperand(0) != c.op->getOperand(0) || a.stride != c.stride ||
        a.dims[0] != c.dims[0] || a.dims[1] != c.dims[1] ||
        a.dims[0] > b.options.max_channels ||
        a.dims[1] > b.options.max_channels)
      return mlir::failure();
    auto input =
        mlir::dyn_cast<mlir::RankedTensorType>(a.op->getOperand(0).getType());
    auto output =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!input || !output || !input.hasStaticShape() ||
        !output.hasStaticShape() || input.getRank() != 4 ||
        output.getRank() != 4 || !input.getElementType().isF32() ||
        !output.getElementType().isF32() || input.getDimSize(1) != a.dims[1] ||
        output.getDimSize(1) != a.dims[0] ||
        a.op->getResult(0).getType() != output ||
        c.op->getResult(0).getType() != output)
      return mlir::failure();
    for (auto d : input.getShape())
      if (d <= 0)
        return mlir::failure();
    if (output.getDimSize(0) != input.getDimSize(0))
      return mlir::failure();
    for (unsigned i = 0; i < 2; ++i)
      if (output.getDimSize(i + 2) !=
          (input.getDimSize(i + 2) - 1) / a.stride[i] + 1)
        return mlir::failure();
    Shape kernel{std::max(a.kernel[0], c.kernel[0]),
                 std::max(a.kernel[1], c.kernel[1])};
    int64_t count = a.dims[0] * a.dims[1] * kernel[0] * kernel[1];
    if (count > 16777216)
      return mlir::failure();
    std::vector<float> weights(count, 0.f), bias(a.dims[0]);
    for (const auto *branch : {&a, &c}) {
      auto dh = (kernel[0] - branch->kernel[0]) / 2,
           dw = (kernel[1] - branch->kernel[1]) / 2;
      for (int64_t channel = 0; channel < a.dims[0] * a.dims[1]; ++channel)
        for (int64_t h = 0; h < branch->kernel[0]; ++h)
          for (int64_t w = 0; w < branch->kernel[1]; ++w)
            weights[(channel * kernel[0] + h + dh) * kernel[1] + w + dw] +=
                branch->weights[(channel * branch->kernel[0] + h) *
                                    branch->kernel[1] +
                                w];
    }
    for (size_t i = 0; i < bias.size(); ++i)
      bias[i] = a.bias[i] + c.bias[i];
    for (auto *data : {&weights, &bias})
      if (!std::all_of(data->begin(), data->end(),
                       [](float x) { return std::isfinite(x); }))
        return mlir::failure();
    auto weight = [&](const Shape &dims, const std::vector<float> &data) {
      auto name = b.uniqueWeightName(id);
      onnx::TensorProto tensor;
      tensor.set_name(name);
      tensor.set_data_type(onnx::TensorProto::FLOAT);
      for (auto d : dims)
        tensor.add_dims(d);
      for (auto x : data)
        tensor.add_float_data(x);
      b.tensors.push_back(std::move(tensor));
      mlir::OperationState state(root->getLoc(), "vbx_onnx.generated_weight");
      state.addTypes(mlir::RankedTensorType::get(dims, rw.getF32Type()));
      return std::make_pair(name, rw.create(state)->getResult(0));
    };
    auto w = weight({a.dims[0], a.dims[1], kernel[0], kernel[1]}, weights);
    auto biasValue = weight({a.dims[0]}, bias);
    onnx::NodeProto node;
    node.set_op_type("Conv");
    node.set_name(add->name() + "_fused_conv");
    node.set_doc_string(add->doc_string());
    node.add_input(a.node->input(0));
    node.add_input(w.first);
    node.add_input(biasValue.first);
    node.add_output(add->output(0));
    auto ints = [&](const char *name, const Shape &values) {
      auto *attr = node.add_attribute();
      attr->set_name(name);
      attr->set_type(onnx::AttributeProto::INTS);
      for (auto x : values)
        attr->add_ints(x);
    };
    ints("kernel_shape", kernel);
    ints("strides", a.stride);
    ints("pads", {kernel[0] / 2, kernel[1] / 2, kernel[0] / 2, kernel[1] / 2});
    mlir::OperationState state(root->getLoc(), "vbx_onnx.lowered_Conv");
    state.addOperands({a.op->getOperand(0), w.second, biasValue.second});
    state.addTypes(output);
    rw.replaceOp(root, rw.create(state)->getResults());
    for (auto *branch : {a.op, c.op}) {
      b.removed.insert(
          branch->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt());
      rw.eraseOp(branch);
    }
    b.replacements.emplace(id, std::vector<onnx::NodeProto>{std::move(node)});
    ++b.stats.rewrites;
    return mlir::success();
  }
};
} // namespace
void populateConvBranchPatterns(mlir::RewritePatternSet &patterns,
                                Bridge &bridge) {
  patterns.add<ConvBranches>(patterns.getContext(), bridge);
}
} // namespace vbx
