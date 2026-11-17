#include "Passes.h"
#include "Graph.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <stdexcept>
namespace vbx {
void populateSdkInverseRegroupPatterns(mlir::RewritePatternSet &, SdkSsaGraph &,
                                       Stats &, const Options &);
void canonicalizeSdkInverseViews(SdkSsaGraph &, Stats &, const Options &);
void writeSdkValue(const llvm::json::Value &value,
                   llvm::json::OStream &stream) {
  if (auto *object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    for (auto &item : *object)
      keys.push_back(item.first);
    // flatc's JSON reader requires union discriminators before union payloads.
    std::sort(keys.begin(), keys.end(), [](auto a, auto b) {
      auto at = a.ends_with("_type"), bt = b.ends_with("_type");
      return at != bt ? at : a < b;
    });
    stream.objectBegin();
    for (auto key : keys) {
      stream.attributeBegin(key);
      writeSdkValue(*object->get(key), stream);
      stream.attributeEnd();
    }
    stream.objectEnd();
  } else if (auto *values = value.getAsArray()) {
    stream.arrayBegin();
    for (auto &item : *values)
      writeSdkValue(item, stream);
    stream.arrayEnd();
  } else
    stream.value(value);
}

namespace {
class RegroupPass final
    : public mlir::PassWrapper<RegroupPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  SdkSsaGraph &graph;
  Stats &stats;
  const Options &options;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RegroupPass)
  RegroupPass(SdkSsaGraph &graph, Stats &stats, const Options &options)
      : graph(graph), stats(stats), options(options) {}
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateSdkInverseRegroupPatterns(patterns, graph, stats, options);
    mlir::GreedyRewriteConfig config;
    config.fold = false;
    config.cseConstants = false;
    config.maxNumRewrites = options.max_rewrites;
    if (mlir::failed(mlir::applyPatternsGreedily(graph.function,
                                                 std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
Stats rewriteSdkJson(std::string &text, const Options &options,
                     std::string *dump) {
  if (options.max_rewrites < 1)
    throw std::runtime_error("max_rewrites must be positive");
  Stats stats;
  std::string regroupDump, layoutDump;
  {
    auto document = llvm::json::parse(text);
    if (!document || !document->getAsObject())
      throw std::runtime_error("Invalid SDK JSON");
    {
      SdkSsaGraph graph(*document->getAsObject());
      stats.nodes = graph.operators->size();
      mlir::PassManager pm(&graph.context);
      pm.enableVerifier(true);
      pm.addPass(std::make_unique<RegroupPass>(graph, stats, options));
      if (mlir::failed(pm.run(*graph.module)))
        throw std::runtime_error("SDK regroup legalization failed");
      canonicalizeSdkInverseViews(graph, stats, options);
      if (dump) {
        llvm::raw_string_ostream stream(regroupDump);
        graph.module->print(stream);
      }
      if (stats.rewrites)
        graph.exportOperators();
    }
    if (stats.rewrites) {
      text.clear();
      llvm::raw_string_ostream stream(text);
      llvm::json::OStream out(stream, 2);
      writeSdkValue(*document, out);
    }
  }
  auto layout =
      rewriteSdkLayoutJson(text, options, dump ? &layoutDump : nullptr);
  stats.candidates += layout.candidates;
  stats.attempts += layout.attempts;
  stats.rewrites += layout.rewrites;
  if (stats.rewrites > size_t(options.max_rewrites))
    throw std::runtime_error("SDK layout rewrite limit exceeded");
  if (dump)
    *dump = regroupDump + layoutDump;
  return stats;
}
} // namespace vbx
