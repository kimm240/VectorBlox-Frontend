#include "Passes.h"
#include "Bridge.h"
#include "Patterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstring>

namespace vbx {
namespace {
class LegalizePass final
    : public mlir::PassWrapper<LegalizePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizePass)
  explicit LegalizePass(Bridge &bridge) : bridge_(&bridge) {}
  LegalizePass(const LegalizePass &other)
      : PassWrapper(other), bridge_(other.bridge_) {}
  llvm::StringRef getArgument() const override { return "vbx-legalize-onnx"; }
  llvm::StringRef getDescription() const override {
    return "Conservative ONNX legalization using upstream MLIR patterns";
  }
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateChannelMeanPatterns(patterns, *bridge_);
    populateUnfoldPatterns(patterns, *bridge_);
    populateLegalizationPatterns(patterns, *bridge_);
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));
    mlir::GreedyRewriteConfig config;
    config.maxNumRewrites = bridge_->options.max_rewrites;
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingAndNewOps;
    config.fold = false;
    config.cseConstants = false;
    // Seed only matching roots. The worklist and SSA use updates are MLIR's.
    if (mlir::failed(
            mlir::applyOpPatternsGreedily(bridge_->candidates, frozen, config)))
      signalPassFailure();
  }

private:
  Bridge *bridge_;
};
} // namespace
std::unique_ptr<mlir::Pass> createLegalizePass(Bridge &bridge) {
  return std::make_unique<LegalizePass>(bridge);
}
} // namespace vbx
