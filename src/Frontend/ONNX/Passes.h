#pragma once
#include <memory>
namespace mlir {
class Pass;
}
namespace vbx {
struct Bridge;
std::unique_ptr<mlir::Pass> createLegalizePass(Bridge &bridge);
} // namespace vbx
