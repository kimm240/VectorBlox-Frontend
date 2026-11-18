#include "Bridge.h"
#include "Patterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace vbx {
namespace {
using Shape = std::vector<int64_t>;
Shape shape(mlir::Value v, bool requireF32 = true) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape() || (requireF32 && !t.getElementType().isF32()))
    return {};
  Shape s(t.getShape().begin(), t.getShape().end());
  for (auto x : s)
    if (x <= 0)
      return {};
  return s;
}
int64_t elements(const Shape &s) {
  int64_t n = 1;
  for (auto x : s) {
    if (x <= 0 || n > INT64_MAX / x)
      return 0;
    n *= x;
  }
  return n;
}
bool inlineTensor(const onnx::TensorProto *t, int type) {
  return t && t->data_type() == type && !t->external_data_size() &&
         t->data_location() != onnx::TensorProto::EXTERNAL;
}
bool readIntTensor(const onnx::TensorProto *t, Shape &out, int64_t limit) {
  if (!inlineTensor(t, onnx::TensorProto::INT64))
    return false;
  auto count = elements(Shape(t->dims().begin(), t->dims().end()));
  if (count < 1 || count > limit)
    return false;
  if (t->has_raw_data()) {
    if (t->raw_data().size() != size_t(count * 8))
      return false;
    for (int64_t i = 0; i < count; ++i) {
      uint64_t u = 0;
      for (int j = 0; j < 8; ++j)
        u |= uint64_t((unsigned char)t->raw_data()[i * 8 + j]) << (8 * j);
      int64_t x;
      std::memcpy(&x, &u, 8);
      out.push_back(x);
    }
  } else
    out.assign(t->int64_data().begin(), t->int64_data().end());
  return out.size() == size_t(count);
}
bool readInts(const onnx::TensorProto *t, Shape &out) {
  return t && t->dims_size() == 1 && readIntTensor(t, out, 16);
}
bool floats(const onnx::TensorProto *t, std::vector<float> &data) {
  if (!inlineTensor(t, onnx::TensorProto::FLOAT))
    return false;
  Shape dims(t->dims().begin(), t->dims().end());
  auto n = elements(dims);
  if (n < 1 || n > 16777216)
    return false;
  data.resize(n);
  if (t->has_raw_data()) {
    if (t->raw_data().size() != size_t(n * 4))
      return false;
    for (int64_t i = 0; i < n; ++i) {
      uint32_t u = 0;
      for (int j = 0; j < 4; ++j)
        u |= uint32_t((unsigned char)t->raw_data()[i * 4 + j]) << (8 * j);
      std::memcpy(&data[i], &u, 4);
    }
  } else {
    if (t->float_data_size() != n)
      return false;
    std::copy(t->float_data().begin(), t->float_data().end(), data.begin());
  }
  return true;
}
bool scalarFloat(Bridge &b, mlir::Value v, float &x) {
  auto *t = b.constants.lookup(v);
  std::vector<float> data;
  if (!floats(t, data) || data.size() != 1 || !std::isfinite(data[0]))
    return false;
  x = data[0];
  return true;
}
const onnx::NodeProto *sourceNode(Bridge &b, mlir::Operation *op,
                                  const char *type) {
  if (!op)
    return nullptr;
  auto id = op->getAttrOfType<mlir::IntegerAttr>("vbx.source");
  if (!id)
    return nullptr;
  auto &n = b.model.graph().node(id.getInt());
  return n.domain().empty() && n.op_type() == type ? &n : nullptr;
}
void attr(onnx::NodeProto &n, const char *key, const Shape &value) {
  auto *a = n.add_attribute();
  a->set_name(key);
  a->set_type(onnx::AttributeProto::INTS);
  for (auto x : value)
    a->add_ints(x);
}
// Fragment builder mirrors every emitted ONNX node in MLIR SSA. New op names
// deliberately do not re-enter legalization; all checks precede construction.
struct Fragment {
  Bridge &b;
  mlir::PatternRewriter &rw;
  mlir::Operation *root;
  int64_t id;
  const onnx::NodeProto &original;
  llvm::StringMap<mlir::Value> values;
  std::vector<onnx::NodeProto> nodes;
  std::map<Shape, std::string> integerNames;
  Fragment(Bridge &b, mlir::PatternRewriter &rw, mlir::Operation *root)
      : b(b), rw(rw), root(root),
        id(root->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt()),
        original(b.model.graph().node(id)) {
    for (int i = 0; i < original.input_size(); ++i)
      if (!original.input(i).empty())
        values[original.input(i)] = root->getOperand(i);
  }
  std::string fresh() { return b.uniqueWeightName(id); }
  std::string constant(const Shape &dims, const std::vector<float> &data) {
    onnx::TensorProto t;
    t.set_name(fresh());
    t.set_data_type(onnx::TensorProto::FLOAT);
    for (auto d : dims)
      t.add_dims(d);
    for (auto x : data)
      t.add_float_data(x);
    auto name = t.name();
    b.tensors.push_back(std::move(t));
    mlir::OperationState s(root->getLoc(), "vbx_onnx.generated_weight");
    s.addTypes(mlir::RankedTensorType::get(dims, rw.getF32Type()));
    values[name] = rw.create(s)->getResult(0);
    return name;
  }
  std::string integers(const Shape &data) {
    auto found = integerNames.find(data);
    if (found != integerNames.end())
      return found->second;
    onnx::TensorProto t;
    t.set_name(fresh());
    t.set_data_type(onnx::TensorProto::INT64);
    t.add_dims(data.size());
    for (auto x : data)
      t.add_int64_data(x);
    auto name = t.name();
    b.tensors.push_back(std::move(t));
    mlir::OperationState s(root->getLoc(), "vbx_onnx.generated_weight");
    s.addTypes(
        mlir::RankedTensorType::get({int64_t(data.size())}, rw.getI64Type()));
    values[name] = rw.create(s)->getResult(0);
    integerNames.emplace(data, name);
    return name;
  }
  std::string
  emit(const char *kind, const std::vector<std::string> &inputs,
       const Shape &dims,
       std::initializer_list<std::pair<const char *, Shape>> attrs = {},
       bool last = false) {
    onnx::NodeProto n;
    n.set_op_type(kind);
    for (auto &x : inputs)
      n.add_input(x);
    auto output = last ? original.output(0) : fresh();
    n.add_output(output);
    n.set_name(output + "_" + kind);
    if (last)
      n.set_doc_string(original.doc_string());
    for (auto &a : attrs)
      attr(n, a.first, a.second);
    mlir::OperationState s(root->getLoc(),
                           std::string("vbx_onnx.lowered_") + kind);
    for (auto &x : inputs)
      s.addOperands(values.lookup(x));
    s.addTypes(mlir::RankedTensorType::get(
        dims, mlir::cast<mlir::RankedTensorType>(root->getResult(0).getType())
                  .getElementType()));
    values[output] = rw.create(s)->getResult(0);
    nodes.push_back(std::move(n));
    return output;
  }
  std::string reshape(const std::string &input, const Shape &dims,
                      bool last = false) {
    auto target = integers(dims);
    return emit("Reshape", {input, target}, dims, {}, last);
  }
  std::vector<std::string> split(const std::string &input, const Shape &dims,
                                 int64_t axis, const Shape &sizes) {
    auto lengths = integers(sizes);
    onnx::NodeProto n;
    n.set_op_type("Split");
    n.add_input(input);
    n.add_input(lengths);
    auto *a = n.add_attribute();
    a->set_name("axis");
    a->set_type(onnx::AttributeProto::INT);
    a->set_i(axis);
    mlir::OperationState state(root->getLoc(), "vbx_onnx.lowered_Split");
    state.addOperands({values.lookup(input), values.lookup(lengths)});
    std::vector<std::string> names;
    for (auto size : sizes) {
      auto shape = dims;
      shape[axis] = size;
      state.addTypes(mlir::RankedTensorType::get(shape, rw.getF32Type()));
      names.push_back(fresh());
      n.add_output(names.back());
    }
    auto *op = rw.create(state);
    for (size_t i = 0; i < names.size(); ++i)
      values[names[i]] = op->getResult(i);
    nodes.push_back(std::move(n));
    return names;
  }
  void finish(const std::string &output) {
    rw.replaceOp(root, values.lookup(output));
    b.replacements.emplace(id, std::move(nodes));
    ++b.stats.rewrites;
  }
};
class Legalize final : public mlir::RewritePattern {
  Bridge &b;

public:
  Legalize(mlir::MLIRContext *c, Bridge &b, const char *name)
      : RewritePattern(name, 1, c), b(b) {
    setDebugName(name);
  }
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rw) const override {
    ++b.stats.attempts;
    auto reject = [&](const char *reason) {
      ++b.stats.skipped[reason];
      return mlir::failure();
    };
    auto &n = b.model.graph().node(
        op->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt());
    if (b.opset < 13 || b.opset > 21 || op->getNumResults() != 1 ||
        n.output_size() != 1 || n.output(0).empty())
      return reject("legalize_opset_or_arity");
    if (op->getNumOperands() < 1)
      return reject("legalize_input_arity");

    auto in = shape(op->getOperand(0)), out = shape(op->getResult(0));
    if (in.empty() || out.empty() || !elements(in) || !elements(out))
      return reject("legalize_requires_static_fp32");
    if (n.op_type() != "Slice" && n.op_type() != "Gather" &&
        n.op_type() != "Reshape" &&
        b.orderSensitive.count(
            op->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt()))
      return reject("arithmetic_feeds_order_sensitive_op");

    if (n.op_type() == "Mul") {
      auto gelu = [&]() -> mlir::LogicalResult {
        // Recognize the exact exported GELU chain, preserving its arithmetic
        // order and constants while tiling independent rows for the SDK LUT.
        if ((in.size() != 3 && in.size() != 4) || in != out ||
            n.attribute_size() || op->getNumOperands() != 2)
          return reject("gelu_shape_or_arity");
        float half;
        if (!scalarFloat(b, op->getOperand(1), half) || half != 0.5f)
          return reject("gelu_final_scale");
        auto *product = op->getOperand(0).getDefiningOp();
        auto *pn = sourceNode(b, product, "Mul");
        if (!pn || pn->attribute_size() || product->getNumOperands() != 2 ||
            !product->getResult(0).hasOneUse())
          return reject("gelu_product");
        auto value = product->getOperand(0);
        auto *add = product->getOperand(1).getDefiningOp();
        auto *an = sourceNode(b, add, "Add");
        float one;
        if (!an || an->attribute_size() || add->getNumOperands() != 2 ||
            !add->getResult(0).hasOneUse() ||
            !scalarFloat(b, add->getOperand(1), one) || one != 1.f)
          return reject("gelu_add");
        auto *erf = add->getOperand(0).getDefiningOp();
        auto *en = sourceNode(b, erf, "Erf");
        if (!en || en->attribute_size() || erf->getNumOperands() != 1 ||
            !erf->getResult(0).hasOneUse())
          return reject("gelu_erf");
        auto *div = erf->getOperand(0).getDefiningOp();
        auto *dn = sourceNode(b, div, "Div");
        float divisor;
        if (!dn || dn->attribute_size() || div->getNumOperands() != 2 ||
            !div->getResult(0).hasOneUse() || div->getOperand(0) != value ||
            !scalarFloat(b, div->getOperand(1), divisor) ||
            divisor != std::sqrt(2.f))
          return reject("gelu_divisor");
        if (shape(value) != in || shape(add->getResult(0)) != in ||
            shape(erf->getResult(0)) != in || shape(div->getResult(0)) != in)
          return reject("gelu_intermediate_shape");
        for (auto &node : b.model.graph().node())
          for (auto &a : node.attribute())
            if (a.type() == onnx::AttributeProto::GRAPH ||
                a.type() == onnx::AttributeProto::GRAPHS)
              return reject("gelu_nested_captures");
        const int64_t tileAxis = in.size() == 4 ? 2 : 1;
        const int64_t perRow = elements(in) / in[tileAxis];
        if (perRow < 1 || perRow > 32768 || (in.size() == 4 && in[0] != 1))
          return reject("gelu_tile_policy");
        auto rows = 32768 / perRow;
        if (in[tileAxis] <= rows || (in[tileAxis] + rows - 1) / rows > 128)
          return reject("gelu_tile_policy");
        Fragment f(b, rw, op);
        f.values[pn->input(0)] = value;
        f.values[dn->input(1)] = div->getOperand(1);
        f.values[an->input(1)] = add->getOperand(1);
        std::vector<std::string> parts;
        for (int64_t begin = 0; begin < in[tileAxis]; begin += rows) {
          auto end = std::min(in[tileAxis], begin + rows);
          auto dims = in;
          dims[tileAxis] = end - begin;
          auto starts = f.integers({begin}), ends = f.integers({end}),
               axes = f.integers({tileAxis});
          auto x = f.emit("Slice", {pn->input(0), starts, ends, axes}, dims);
          auto d = f.emit("Div", {x, dn->input(1)}, dims);
          auto e = f.emit("Erf", {d}, dims);
          auto a = f.emit("Add", {e, an->input(1)}, dims);
          auto m = f.emit("Mul", {x, a}, dims);
          parts.push_back(f.emit("Mul", {m, n.input(1)}, dims));
        }
        auto result = f.emit("Concat", parts, out, {}, true);
        auto *axis = f.nodes.back().add_attribute();
        axis->set_name("axis");
        axis->set_type(onnx::AttributeProto::INT);
        axis->set_i(tileAxis);
        f.finish(result);
        for (auto *parent : {product, add, erf, div}) {
          b.removed.insert(
              parent->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt());
          rw.eraseOp(parent);
        }
        return mlir::success();
      };
      // Preserve complete GELU chains before considering a scalar Conv, so
      // the downstream converter can still fuse Erf into native GELU.
      if (mlir::succeeded(gelu()))
        return mlir::success();
      // A large scalar scale after QKV transpose may become an SDK LUT
      // sublayer with insufficient scratchpad. A one-tap depthwise Conv
      // performs precisely one multiplication per element without mixing
      // channels or materializing a diagonal weight matrix.
      if (in.size() == 4 && in == out && in[0] == 1 &&
          in[1] <= b.options.max_channels && elements(in) > 32768 &&
          n.attribute_size() == 0 && op->getNumOperands() == 2) {
        float coefficient;
        if (scalarFloat(b, op->getOperand(1), coefficient) &&
            std::isnormal(coefficient) && coefficient != 1.f) {
          Fragment f(b, rw, op);
          auto weight = f.constant({in[1], 1, 1, 1},
                                   std::vector<float>(in[1], coefficient));
          auto result = f.emit("Conv", {n.input(0), weight}, out,
                               {{"kernel_shape", {1, 1}}}, true);
          auto *group = f.nodes.back().add_attribute();
          group->set_name("group");
          group->set_type(onnx::AttributeProto::INT);
          group->set_i(in[1]);
          f.finish(result);
          return mlir::success();
        }
      }
      return mlir::failure();
    }
    if (n.op_type() == "Softmax") {
      if (in.size() != 4 || in[0] != 1 || in != out || n.input_size() != 1 ||
          n.attribute_size() > 1)
        return reject("softmax_requires_rank4");
      int64_t axis = -1;
      if (n.attribute_size()) {
        auto &a = n.attribute(0);
        if (a.name() != "axis" || a.type() != onnx::AttributeProto::INT)
          return reject("softmax_axis_attribute");
        axis = a.i();
      }
      if (axis != -1 && axis != 3)
        return reject("softmax_requires_last_axis");
      auto K = in.back();
      if (K <= 256 || K > b.options.max_channels || in[1] > 64)
        return reject("softmax_tile_policy");
      // Bound the exp/sum/output working set while retaining rank-four layout.
      // This is an explicit frontend tiling policy, not a hardware-size claim.
      auto tile = 32768 / (6 * K * in[1]);
      if (tile < 1 || tile >= in[2] || (in[2] + tile - 1) / tile > 128)
        return reject("softmax_tile_policy");
      Fragment f(b, rw, op);
      std::vector<std::string> parts;
      for (int64_t begin = 0; begin < in[2]; begin += tile) {
        auto end = std::min(in[2], begin + tile);
        Shape part = in;
        part[2] = end - begin;
        auto starts = f.integers({begin}), ends = f.integers({end}),
             axes = f.integers({2});
        auto x = f.emit("Slice", {n.input(0), starts, ends, axes}, part);
        parts.push_back(f.emit("Softmax", {x}, part));
      }
      auto x = f.emit("Concat", parts, out, {}, true);
      auto *axisAttr = f.nodes.back().add_attribute();
      axisAttr->set_name("axis");
      axisAttr->set_type(onnx::AttributeProto::INT);
      axisAttr->set_i(2);
      f.finish(x);
      return mlir::success();
    }

    if (n.op_type() == "LayerNormalization") {
      if (in.size() != 3 || in != out || n.input_size() < 2 ||
          n.input_size() > 3 || b.opset < 17)
        return reject("layernorm_rank_or_arity");
      int64_t axis = -1;
      float epsilon = 1e-5f;
      std::set<std::string> seen;
      for (auto &a : n.attribute()) {
        if (!seen.insert(a.name()).second)
          return reject("layernorm_duplicate_attribute");
        if (a.name() == "axis" && a.type() == onnx::AttributeProto::INT)
          axis = a.i();
        else if (a.name() == "epsilon" &&
                 a.type() == onnx::AttributeProto::FLOAT)
          epsilon = a.f();
        else if (a.name() == "stash_type" &&
                 a.type() == onnx::AttributeProto::INT && a.i() == 1) {
        } else
          return reject("layernorm_attribute");
      }
      auto K = in.back(), M = elements(in) / K;
      if ((axis != -1 && axis != 2) || !std::isfinite(epsilon) ||
          epsilon <= 0 || K > b.options.max_channels || M > 1048576)
        return reject("layernorm_axis_or_policy");
      if (shape(op->getOperand(1)) != Shape{K})
        return reject("layernorm_scale_shape");
      bool bias = n.input_size() == 3 && !n.input(2).empty();
      if (bias && shape(op->getOperand(2)) != Shape{K})
        return reject("layernorm_bias_shape");
      // Move centered-square after the layout conversion: SDK 3.1 fails
      // allocating the fused square-LUT -> rank-changing Reshape subgraph.
      // Keep the surrounding rank-three arithmetic layout unchanged.
      // Limit each independent row group to 8192 elements. This bounds
      // broadcast intermediates in SDK fused normalization sublayers.
      if (in[0] * K > 8192)
        return reject("layernorm_tile_policy");
      auto rows = 8192 / (in[0] * K);
      if ((in[1] + rows - 1) / rows > 128)
        return reject("layernorm_tile_policy");
      Fragment f(b, rw, op);
      auto weights = f.constant({1, K, 1, 1}, std::vector<float>(K, 1.f));
      auto divisor = f.constant({}, std::vector<float>{float(K)});
      auto normalize = [&](const std::string &input, const Shape &dims,
                           bool last) {
        auto count = elements(dims) / K;
        Shape layout{1, K, count, 1}, convReduced{1, 1, count, 1},
            reduced = dims;
        reduced.back() = 1;
        auto toConv = [&](const std::string &value) {
          auto x = f.reshape(value, {1, count, K, 1});
          return f.emit("Transpose", {x}, layout, {{"perm", {0, 2, 1, 3}}});
        };
        // Sum then divide to preserve constant rows at K=384.
        auto mean = [&](const std::string &value) {
          auto sum = f.emit("Conv", {value, weights}, convReduced,
                            {{"kernel_shape", {1, 1}}});
          sum = f.emit("Div", {sum, divisor}, convReduced);
          return f.reshape(sum, reduced);
        };
        auto mu = mean(toConv(input));
        auto centered = f.emit("Sub", {input, mu}, dims);
        auto convCentered = toConv(centered);
        auto square = f.emit("Mul", {convCentered, convCentered}, layout);
        auto variance = mean(square);
        auto eps = f.constant({}, std::vector<float>{epsilon});
        auto denom = f.emit("Add", {variance, eps}, reduced);
        denom = f.emit("Sqrt", {denom}, reduced);
        auto norm = f.emit("Div", {centered, denom}, dims);
        auto scaled =
            f.emit("Mul", {norm, n.input(1)}, dims, {}, last && !bias);
        if (bias)
          scaled = f.emit("Add", {scaled, n.input(2)}, dims, {}, last);
        return scaled;
      };
      if (in[1] <= rows) {
        auto result = normalize(n.input(0), in, true);
        f.finish(result);
        return mlir::success();
      }
      std::vector<std::string> parts;
      for (int64_t begin = 0; begin < in[1]; begin += rows) {
        auto end = std::min(in[1], begin + rows);
        auto dims = in;
        dims[1] = end - begin;
        auto start = f.integers({begin}), stop = f.integers({end}),
             axes = f.integers({1});
        auto x = f.emit("Slice", {n.input(0), start, stop, axes}, dims);
        parts.push_back(normalize(x, dims, false));
      }
      auto result = f.emit("Concat", parts, out, {}, true);
      auto *axisAttr = f.nodes.back().add_attribute();
      axisAttr->set_name("axis");
      axisAttr->set_type(onnx::AttributeProto::INT);
      axisAttr->set_i(1);
      f.finish(result);
      return mlir::success();
    }
    if (n.op_type() == "Gather") {
      if ((in.size() != 5 && in.size() != 6) || n.input_size() != 2 ||
          n.attribute_size() > 1 || elements(in) > 16777216)
        return reject("gather_requires_rank5_or_rank6_scalar");
      int64_t axis = 0;
      if (n.attribute_size()) {
        auto &a = n.attribute(0);
        if (a.name() != "axis" || a.type() != onnx::AttributeProto::INT)
          return reject("gather_axis_attribute");
        axis = a.i();
      }
      if (axis < 0)
        axis += in.size();
      if (axis < 0 || axis >= int64_t(in.size()))
        return reject("gather_invalid_axis");
      auto *t = b.constants.lookup(op->getOperand(1));
      if (!inlineTensor(t, onnx::TensorProto::INT64) || t->dims_size() != 0)
        return reject("gather_requires_constant_scalar_index");
      onnx::TensorProto scalar(*t);
      scalar.add_dims(1);
      Shape index;
      if (!readInts(&scalar, index))
        return reject("gather_invalid_index");
      int64_t i = index[0];
      if (i < 0)
        i += in[axis];
      if (i < 0 || i >= in[axis])
        return reject("gather_index_out_of_bounds");
      Shape expected = in;
      expected.erase(expected.begin() + axis);
      if (out != expected)
        return reject("gather_output_shape");
      if (in.size() == 6 || axis != 0) {
        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < axis; ++d)
          outer *= in[d];
        for (size_t d = axis + 1; d < in.size(); ++d)
          inner *= in[d];
        Fragment f(b, rw, op);
        auto flat = f.reshape(n.input(0), {1, outer, in[axis], inner});
        auto start = f.integers({i}), end = f.integers({i + 1}),
             axes = f.integers({2});
        auto selected =
            f.emit("Slice", {flat, start, end, axes}, {1, outer, 1, inner});
        f.finish(f.reshape(selected, out, true));
        return mlir::success();
      }
      // Split packed heads before introducing the unsupported rank-five tensor.
      auto *transpose = op->getOperand(0).getDefiningOp();
      auto *tn = sourceNode(b, transpose, "Transpose");
      auto *reshape = tn && transpose->getNumOperands() == 1
                          ? transpose->getOperand(0).getDefiningOp()
                          : nullptr;
      auto *rn = sourceNode(b, reshape, "Reshape");
      bool nested = false;
      for (auto &node : b.model.graph().node())
        for (auto &a : node.attribute())
          nested |= a.type() == onnx::AttributeProto::GRAPH ||
                    a.type() == onnx::AttributeProto::GRAPHS;
      if (!nested && tn && rn && tn->attribute_size() == 1 &&
          tn->attribute(0).name() == "perm" &&
          tn->attribute(0).type() == onnx::AttributeProto::INTS &&
          Shape(tn->attribute(0).ints().begin(),
                tn->attribute(0).ints().end()) == Shape{2, 0, 3, 1, 4} &&
          (rn->attribute_size() == 0 ||
           (rn->attribute_size() == 1 &&
            rn->attribute(0).name() == "allowzero" &&
            rn->attribute(0).type() == onnx::AttributeProto::INT &&
            (rn->attribute(0).i() == 0 || rn->attribute(0).i() == 1))) &&
          reshape->getNumOperands() == 2) {
        auto base = shape(reshape->getOperand(0)),
             packed = shape(reshape->getResult(0));
        Shape target;
        if (base.size() == 3 && base[0] == in[1] && base[1] == in[3] &&
            base[2] == in[0] * in[2] * in[4] &&
            packed == Shape{in[1], in[3], in[0], in[2], in[4]} &&
            readInts(b.constants.lookup(reshape->getOperand(1)), target) &&
            target == packed) {
          Fragment f(b, rw, op);
          f.values[rn->input(0)] = reshape->getOperand(0);
          auto width = in[2] * in[4];
          auto start = f.integers({i * width}),
               end = f.integers({(i + 1) * width}), axes = f.integers({2});
          auto x = f.emit("Slice", {rn->input(0), start, end, axes},
                          {base[0], base[1], width});
          x = f.reshape(x, {base[0], base[1], in[2], in[4]});
          x = f.emit("Transpose", {x}, out, {{"perm", {0, 2, 1, 3}}}, true);
          f.finish(x);
          if (transpose->getResult(0).use_empty()) {
            b.removed.insert(
                transpose->getAttrOfType<mlir::IntegerAttr>("vbx.source")
                    .getInt());
            rw.eraseOp(transpose);
            if (reshape->getResult(0).use_empty()) {
              b.removed.insert(
                  reshape->getAttrOfType<mlir::IntegerAttr>("vbx.source")
                      .getInt());
              rw.eraseOp(reshape);
            }
          }
          return mlir::success();
        }
      }
      Fragment f(b, rw, op);
      auto start = f.integers({i}), end = f.integers({i + 1}),
           axes = f.integers({0});
      auto sliced = in;
      sliced[0] = 1;
      auto x = f.emit("Slice", {n.input(0), start, end, axes}, sliced);
      x = f.reshape(x, out, true);
      f.finish(x);
      return mlir::success();
    }

    if (n.op_type() == "MatMul" || n.op_type() == "Gemm") {
      bool gemm = n.op_type() == "Gemm";
      int64_t transB = 0;
      if ((!gemm && (n.attribute_size() || n.input_size() != 2)) ||
          (gemm &&
           (in.size() != 2 || n.input_size() < 2 || n.input_size() > 3)))
        return reject("linear_arity_or_attributes");
      std::set<std::string> seen;
      for (auto &a : n.attribute()) {
        if (!seen.insert(a.name()).second)
          return reject("linear_duplicate_attribute");
        if (a.name() == "transB" && a.type() == onnx::AttributeProto::INT &&
            (a.i() == 0 || a.i() == 1))
          transB = a.i();
        else if (a.name() == "transA" &&
                 a.type() == onnx::AttributeProto::INT && a.i() == 0) {
        } else if ((a.name() == "alpha" || a.name() == "beta") &&
                   a.type() == onnx::AttributeProto::FLOAT && a.f() == 1.f) {
        } else
          return reject("linear_attributes_not_supported");
      }
      auto right = shape(op->getOperand(1));
      if (!gemm && in.size() == 4 && right.size() == 4) {
        auto K = in[3];
        if (in[0] != 1 || right[0] != in[0] || right[1] != in[1] ||
            right[2] != K || in[1] > 64 || K > 1024 ||
            (K <= 128 && elements(in) <= 32768) || K * right[3] > 65536 ||
            out != Shape{in[0], in[1], in[2], right[3]} ||
            elements(out) > 16777216 || elements(out) * K > 33554432)
          return reject("batch_matmul_outer_product_policy");
        // Express each independent head as broadcast products and a constant
        // channel-sum Conv, avoiding the SDK dynamic BATCH_MATMUL allocator.
        // Cap the total product tensor volume to bound this fallback's cost.
        Fragment f(b, rw, op);
        Shape sizes(in[1], 1);
        auto left = f.split(n.input(0), in, 1, sizes),
             rhs = f.split(n.input(1), right, 1, sizes);
        auto single = out;
        single[1] = 1;
        std::vector<std::string> heads;
        auto weights = f.constant({1, K, 1, 1}, std::vector<float>(K, 1.f));
        for (size_t i = 0; i < sizes.size(); ++i) {
          auto a = f.emit("Transpose", {left[i]}, {1, K, in[2], 1},
                          {{"perm", {0, 3, 2, 1}}});
          auto bvalue = f.emit("Transpose", {rhs[i]}, {1, K, 1, right[3]},
                               {{"perm", {0, 2, 1, 3}}});
          auto product = f.emit("Mul", {a, bvalue}, {1, K, in[2], right[3]});
          heads.push_back(f.emit("Conv", {product, weights}, single,
                                 {{"kernel_shape", {1, 1}}}));
        }
        auto x = f.emit("Concat", heads, out, {}, true);
        auto *axis = f.nodes.back().add_attribute();
        axis->set_name("axis");
        axis->set_type(onnx::AttributeProto::INT);
        axis->set_i(1);
        f.finish(x);
        return mlir::success();
      }
      auto *weight = b.constants.lookup(op->getOperand(1));
      std::vector<float> data;
      if (!weight || weight->dims_size() != 2 || !floats(weight, data))
        return reject("linear_requires_inline_constant_weights");
      auto K = in.back(), N = weight->dims(transB ? 0 : 1),
           M = elements(in) / K;
      if (in.size() < 2 || in.size() > 4 || K != weight->dims(transB ? 1 : 0) ||
          N < 1 || K > b.options.max_channels || N > b.options.max_channels ||
          M > 1048576)
        return reject("linear_shape_or_policy");
      auto expected = in;
      expected.back() = N;
      if (out != expected)
        return reject("linear_output_shape");
      bool bias = gemm && n.input_size() == 3 && !n.input(2).empty();
      if (bias) {
        auto bs = shape(op->getOperand(2));
        if (bs != Shape{N} && bs != Shape{1, N})
          return reject("linear_bias_shape");
      }
      // Fold a private flatten of the last two axes into the convolution
      // kernel. This avoids SDK reshape fusion that confuses head merging with
      // expand_dims.
      auto *flatten = op->getOperand(0).getDefiningOp();
      auto *fn = sourceNode(b, flatten, "Reshape");
      bool nested = false;
      for (auto &node : b.model.graph().node())
        for (auto &at : node.attribute())
          nested |= at.type() == onnx::AttributeProto::GRAPH ||
                    at.type() == onnx::AttributeProto::GRAPHS;
      if (!gemm && in.size() == 3 && in[0] == 1 && fn && !nested &&
          flatten->getNumOperands() == 2 && flatten->getResult(0).hasOneUse()) {
        auto base = shape(flatten->getOperand(0));
        Shape target;
        bool attrs = fn->attribute_size() == 0 ||
                     (fn->attribute_size() == 1 &&
                      fn->attribute(0).name() == "allowzero" &&
                      fn->attribute(0).type() == onnx::AttributeProto::INT &&
                      fn->attribute(0).i() == 0);
        if (attrs && base.size() == 4 && elements(base) && base[0] == 1 &&
            base[1] == in[1] && base[2] * base[3] == K && base[2] <= 64 &&
            readInts(b.constants.lookup(flatten->getOperand(1)), target) &&
            target == in) {
          auto H = base[2], D = base[3];
          std::vector<float> kernel(N * K);
          for (int64_t j = 0; j < N; ++j)
            for (int64_t d = 0; d < D; ++d)
              for (int64_t h = 0; h < H; ++h)
                kernel[(j * D + d) * H + h] = data[(h * D + d) * N + j];
          Fragment f(b, rw, op);
          f.values[fn->input(0)] = flatten->getOperand(0);
          auto weights = f.constant({N, D, 1, H}, kernel);
          auto x = f.emit("Transpose", {fn->input(0)}, {1, D, M, H},
                          {{"perm", {0, 3, 1, 2}}});
          x = f.emit("Conv", {x, weights}, {1, N, M, 1},
                     {{"kernel_shape", {1, H}}});
          x = f.emit("Transpose", {x}, {1, M, N, 1}, {{"perm", {0, 2, 1, 3}}});
          x = f.reshape(x, out, true);
          f.finish(x);
          b.removed.insert(
              flatten->getAttrOfType<mlir::IntegerAttr>("vbx.source").getInt());
          rw.eraseOp(flatten);
          return mlir::success();
        }
      }
      std::vector<float> w(N * K);
      for (int64_t j = 0; j < N; ++j)
        for (int64_t i = 0; i < K; ++i)
          w[j * K + i] = data[transB ? j * K + i : i * N + j];
      Fragment f(b, rw, op);
      auto weights = f.constant({N, K, 1, 1}, w);
      auto x = f.reshape(n.input(0), {1, M, K, 1});
      x = f.emit("Transpose", {x}, {1, K, M, 1}, {{"perm", {0, 2, 1, 3}}});
      x = f.emit("Conv", {x, weights}, {1, N, M, 1},
                 {{"kernel_shape", {1, 1}}});
      x = f.emit("Transpose", {x}, {1, M, N, 1}, {{"perm", {0, 2, 1, 3}}});
      x = f.reshape(x, out, !bias);
      if (bias)
        x = f.emit("Add", {x, n.input(2)}, out, {}, true);
      f.finish(x);
      return mlir::success();
    }
    if (n.op_type() == "ConvTranspose") {
      if (in.size() != 4 || n.input_size() < 2 || n.input_size() > 3)
        return reject("deconv_arity_or_rank");
      std::set<std::string> seen;
      bool stride = false;
      for (auto &a : n.attribute()) {
        if (!seen.insert(a.name()).second)
          return reject("deconv_duplicate_attribute");
        Shape v(a.ints().begin(), a.ints().end());
        if (a.name() == "strides" && a.type() == onnx::AttributeProto::INTS &&
            v == Shape{4, 4})
          stride = true;
        else if (a.name() == "kernel_shape" &&
                 a.type() == onnx::AttributeProto::INTS && v == Shape{4, 4}) {
        } else if (a.name() == "pads" &&
                   a.type() == onnx::AttributeProto::INTS &&
                   v == Shape{0, 0, 0, 0}) {
        } else if (a.name() == "dilations" &&
                   a.type() == onnx::AttributeProto::INTS && v == Shape{1, 1}) {
        } else if (a.name() == "output_padding" &&
                   a.type() == onnx::AttributeProto::INTS && v == Shape{0, 0}) {
        } else if (a.name() == "group" &&
                   a.type() == onnx::AttributeProto::INT && a.i() == 1) {
        } else if (a.name() == "auto_pad" &&
                   a.type() == onnx::AttributeProto::STRING &&
                   a.s() == "NOTSET") {
        } else
          return reject("deconv_requires_nonoverlap_stride4");
      }
      auto *weight = b.constants.lookup(op->getOperand(1));
      std::vector<float> data;
      if (!stride || !weight || weight->dims_size() != 4 ||
          weight->dims(0) != in[1] || weight->dims(2) != 4 ||
          weight->dims(3) != 4 || !floats(weight, data))
        return reject("deconv_weights_or_stride");
      auto C = in[1], N = weight->dims(1);
      if (N < 1 || N > 256 || C > 4096 || in[2] > INT64_MAX / 4 ||
          in[3] > INT64_MAX / 4 || out != Shape{in[0], N, in[2] * 4, in[3] * 4})
        return reject("deconv_shape_or_policy");
      bool bias = n.input_size() == 3 && !n.input(2).empty();
      if (bias && shape(op->getOperand(2)) != Shape{N})
        return reject("deconv_bias_shape");
      std::vector<float> w1(C * N * 16), w2(N * 4 * N * 4, 0.f);
      for (int64_t ci = 0; ci < C; ++ci)
        for (int64_t co = 0; co < N; ++co)
          for (int64_t y = 0; y < 4; ++y)
            for (int64_t x = 0; x < 4; ++x)
              w1[((ci * (N * 4) + co * 4 + (y % 2) * 2 + x % 2) * 2 + y / 2) *
                     2 +
                 x / 2] = data[((ci * N + co) * 4 + y) * 4 + x];
      for (int64_t co = 0; co < N; ++co)
        for (int64_t j = 0; j < 4; ++j)
          w2[((co * 4 + j) * N + co) * 4 + j] = 1.f;
      Fragment f(b, rw, op);
      auto a = f.constant({C, N * 4, 2, 2}, w1),
           c = f.constant({N * 4, N, 2, 2}, w2);
      auto x = f.emit("ConvTranspose", {n.input(0), a},
                      {in[0], N * 4, in[2] * 2, in[3] * 2},
                      {{"kernel_shape", {2, 2}}, {"strides", {2, 2}}});
      std::vector<std::string> ins{x, c};
      if (bias)
        ins.push_back(n.input(2));
      x = f.emit("ConvTranspose", ins, out,
                 {{"kernel_shape", {2, 2}}, {"strides", {2, 2}}}, true);
      f.finish(x);
      return mlir::success();
    }
    if (n.op_type() == "Slice") {
      if (in.size() <= 5 || in.size() > 8 || n.attribute_size() ||
          n.input_size() < 4 || n.input_size() > 5)
        return reject("slice_rank_or_arity");
      Shape starts, ends, axes, steps;
      if (!readInts(b.constants.lookup(op->getOperand(1)), starts) ||
          !readInts(b.constants.lookup(op->getOperand(2)), ends) ||
          !readInts(b.constants.lookup(op->getOperand(3)), axes))
        return reject("slice_constant_parameters");
      if (n.input_size() == 5 &&
          !readInts(b.constants.lookup(op->getOperand(4)), steps))
        return reject("slice_constant_steps");
      if (steps.empty())
        steps.assign(axes.size(), 1);
      if (starts.size() != axes.size() || ends.size() != axes.size() ||
          steps.size() != axes.size())
        return reject("slice_parameter_lengths");
      Shape reduced, mapping(in.size(), -1);
      for (size_t i = 0; i < in.size(); ++i)
        if (in[i] != 1) {
          mapping[i] = reduced.size();
          reduced.push_back(in[i]);
        }
      if (reduced.size() < 2 || reduced.size() > 4)
        return reject("slice_cannot_reduce_rank");
      Shape expected = in, newaxes;
      std::set<int64_t> seen;
      for (size_t j = 0; j < axes.size(); ++j) {
        auto axis = axes[j] < 0 ? axes[j] + int64_t(in.size()) : axes[j];
        if (axis < 0 || axis >= int64_t(in.size()) || mapping[axis] < 0 ||
            steps[j] != 1 || !seen.insert(axis).second)
          return reject("slice_axis_or_step");
        auto norm = [&](int64_t x) {
          return std::max(int64_t(0),
                          std::min(in[axis], x < 0 ? x + in[axis] : x));
        };
        expected[axis] = std::max(int64_t(0), norm(ends[j]) - norm(starts[j]));
        newaxes.push_back(mapping[axis]);
      }
      if (expected != out || !elements(expected))
        return reject("slice_output_shape");
      Shape reducedOut;
      for (size_t i = 0; i < in.size(); ++i)
        if (mapping[i] >= 0)
          reducedOut.push_back(out[i]);
      Fragment f(b, rw, op);
      auto x = f.reshape(n.input(0), reduced);
      auto ax = f.integers(newaxes);
      std::vector<std::string> ins{x, n.input(1), n.input(2), ax};
      if (n.input_size() == 5)
        ins.push_back(n.input(4));
      x = f.emit("Slice", ins, reducedOut);
      x = f.reshape(x, out, true);
      f.finish(x);
      return mlir::success();
    }
    return reject("unknown_legalization");
  }
};
} // namespace
void populateLegalizationPatterns(mlir::RewritePatternSet &patterns,
                                  Bridge &bridge) {
  for (auto *name :
       {"vbx_onnx.MatMul", "vbx_onnx.Gemm", "vbx_onnx.ConvTranspose",
        "vbx_onnx.Slice", "vbx_onnx.Gather", "vbx_onnx.LayerNormalization",
        "vbx_onnx.Softmax", "vbx_onnx.Mul"})
    patterns.add<Legalize>(patterns.getContext(), bridge, name);
}
} // namespace vbx
