#include "Bridge.h"
#include "Patterns.h"
#include <cstring>

namespace vbx {
namespace {
class ChannelMean final : public mlir::RewritePattern {
public:
  ChannelMean(mlir::MLIRContext *context, Bridge &bridge)
      : RewritePattern("vbx_onnx.ReduceMean", 1, context), bridge_(bridge) {
    setDebugName("ChannelMeanToPointwiseConv");
  }
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *root,
                  mlir::PatternRewriter &rw) const override {
    ++bridge_.stats.attempts;
    auto reject = [&](const char *reason) {
      ++bridge_.stats.skipped[reason];
      return rw.notifyMatchFailure(root, reason);
    };
    if (bridge_.opset < 11 || bridge_.opset > 21)
      return reject("opset_outside_tested_11_to_21");
    int64_t id = root->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt();
    if (bridge_.orderSensitive.count(id))
      return reject("arithmetic_feeds_order_sensitive_op");
    const auto &node = bridge_.model.graph().node(static_cast<int>(id));
    if (node.input_size() < 1 || node.input(0).empty() ||
        node.output_size() != 1 || node.output(0).empty())
      return reject("invalid_reduce_mean_arity");
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getOperand(0).getType());
    if (!type || type.getRank() != 4)
      return reject("requires_rank_four");
    if (!type.getElementType().isF32())
      return reject("requires_fp32");
    if (!type.hasStaticShape())
      return reject("requires_static_shape");
    for (auto d : type.getShape())
      if (d <= 0)
        return reject("requires_positive_dimensions");
    int64_t channels = type.getDimSize(1);
    if (channels > bridge_.options.max_channels)
      return reject("channel_policy_limit");
    const onnx::AttributeProto *axes = nullptr;
    bool seenKeep = false, seenNoop = false;
    for (auto &a : node.attribute()) {
      if (a.name() == "keepdims") {
        if (seenKeep || a.type() != onnx::AttributeProto::INT || a.i() != 1)
          return reject("requires_keepdims_one");
        seenKeep = true;
      } else if (a.name() == "axes") {
        if (axes || bridge_.opset >= 18 ||
            a.type() != onnx::AttributeProto::INTS)
          return reject("invalid_axes_attribute");
        axes = &a;
      } else if (a.name() == "noop_with_empty_axes") {
        if (seenNoop || bridge_.opset < 18 ||
            a.type() != onnx::AttributeProto::INT || (a.i() != 0 && a.i() != 1))
          return reject("invalid_noop_attribute");
        seenNoop = true;
      } else
        return reject("unrecognized_reduce_mean_attribute");
    }
    int64_t axis = 0;
    if (bridge_.opset < 18) {
      if (node.input_size() != 1 || !axes || axes->ints_size() != 1)
        return reject("requires_single_explicit_axis");
      axis = axes->ints(0);
    } else {
      if (node.input_size() != 2 || node.input(1).empty())
        return reject("requires_constant_axes_input");
      auto *tensor = bridge_.constants.lookup(root->getOperand(1));
      if (!tensor || tensor->data_type() != onnx::TensorProto::INT64 ||
          tensor->dims_size() != 1 || tensor->dims(0) != 1 ||
          tensor->data_location() == onnx::TensorProto::EXTERNAL ||
          tensor->external_data_size())
        return reject("requires_single_inline_int64_axis");
      if (tensor->has_raw_data()) {
        if (tensor->raw_data().size() != 8)
          return reject("invalid_axis_data");
        uint64_t bits = 0;
        for (unsigned i = 0; i < 8; ++i)
          bits |= uint64_t(static_cast<unsigned char>(tensor->raw_data()[i]))
                  << (8 * i);
        std::memcpy(&axis, &bits, sizeof(axis));
      } else {
        if (tensor->int64_data_size() != 1)
          return reject("invalid_axis_data");
        axis = tensor->int64_data(0);
      }
    }
    if (axis != 1 && axis != -3)
      return reject("not_channel_axis");
    // Match complete: all IR mutation below goes through the upstream rewriter.
    auto name = bridge_.uniqueWeightName(id);
    mlir::OperationState weightState(root->getLoc(),
                                     "vbx_onnx.generated_weight");
    weightState.addTypes(
        mlir::RankedTensorType::get({1, channels, 1, 1}, rw.getF32Type()));
    weightState.addAttribute("vbx.name", rw.getStringAttr(name));
    auto *weight = rw.create(weightState);
    mlir::OperationState convState(root->getLoc(), "vbx_onnx.Conv");
    convState.addOperands({root->getOperand(0), weight->getResult(0)});
    convState.addTypes(root->getResultTypes());
    convState.addAttribute("vbx.source", rw.getI64IntegerAttr(id));
    convState.addAttribute("vbx.generated_weight", rw.getStringAttr(name));
    auto *conv = rw.create(convState);
    rw.replaceOp(root, conv->getResults());
    bridge_.generated.push_back({std::move(name), channels});
    ++bridge_.stats.rewrites;
    return mlir::success();
  }

private:
  Bridge &bridge_;
};
} // namespace

void populateChannelMeanPatterns(mlir::RewritePatternSet &patterns,
                                 Bridge &bridge) {
  patterns.add<ChannelMean>(patterns.getContext(), bridge);
}
} // namespace vbx
