#pragma once
#include "vbx_rewrite/engine.h"
#include "llvm/Support/JSON.h"
namespace vbx {
void writeSdkValue(const llvm::json::Value &, llvm::json::OStream &);
Stats rewriteSdkLayoutJson(std::string &, const Options &, std::string *);
} // namespace vbx
