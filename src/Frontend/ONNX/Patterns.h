#pragma once
namespace mlir {
class RewritePatternSet;
}
namespace vbx {
struct Bridge;
void populateChannelMeanPatterns(mlir::RewritePatternSet &, Bridge &);
void populateUnfoldPatterns(mlir::RewritePatternSet &, Bridge &);
void populateLegalizationPatterns(mlir::RewritePatternSet &, Bridge &);
} // namespace vbx
