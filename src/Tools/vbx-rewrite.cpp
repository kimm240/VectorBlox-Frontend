#include "vbx_rewrite/engine.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
std::string quoted(const std::string &s) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
          << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
void writeText(const std::string &path, const std::string &text) {
  std::ofstream out(path);
  if (!out || !(out << text))
    throw std::runtime_error("Cannot write " + path);
}
void writeTextAtomic(const fs::path &path, const std::string &text) {
  std::string temp = path.string() + ".tmp.XXXXXX";
  int fd = mkstemp(temp.data());
  if (fd < 0)
    throw std::runtime_error("Cannot create output temporary file");
  close(fd);
  try {
    std::ofstream out(temp);
    out << text;
    out.close();
    if (!out)
      throw std::runtime_error("JSON output flush failed");
    fs::rename(temp, path);
  } catch (...) {
    fs::remove(temp);
    throw;
  }
}
void writeModel(const fs::path &path, const onnx::ModelProto &model) {
  std::string temp = path.string() + ".tmp.XXXXXX";
  int fd = mkstemp(temp.data());
  if (fd < 0)
    throw std::runtime_error("Cannot create output temporary file");
  close(fd);
  try {
    std::ofstream out(temp, std::ios::binary);
    if (!model.SerializeToOstream(&out))
      throw std::runtime_error("ONNX serialization failed");
    out.close();
    if (!out)
      throw std::runtime_error("ONNX output flush failed");
    fs::rename(temp, path);
  } catch (...) {
    fs::remove(temp);
    throw;
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    auto totalStart = Clock::now();
    std::string input, output, report, dump;
    bool sdkJson = false;
    vbx::Options options;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--help" || a == "-h") {
        std::cout
            << "vbx-rewrite INPUT OUTPUT [--report FILE] [--dump-mlir FILE]\n"
            << "  --sdk-json: SDK flatc JSON layout legalization\n"
            << "  --max-channels N: ONNX channel policy limit\n"
            << "  --max-rewrites N: maximum rewrite count\n";
        return 0;
      }
      auto value = [&]() {
        if (++i >= argc)
          throw std::runtime_error("Missing value for " + a);
        return std::string(argv[i]);
      };
      if (a == "--sdk-json")
        sdkJson = true;
      else if (a == "--report")
        report = value();
      else if (a == "--dump-mlir")
        dump = value();
      else if (a == "--max-channels")
        options.max_channels = std::stoll(value());
      else if (a == "--max-rewrites")
        options.max_rewrites = std::stoll(value());
      else if (!a.empty() && a[0] == '-')
        throw std::runtime_error("Unknown option: " + a);
      else if (input.empty())
        input = a;
      else if (output.empty())
        output = a;
      else
        throw std::runtime_error("Unexpected positional argument");
    }
    if (input.empty() || output.empty())
      throw std::runtime_error(
          "Usage: vbx-rewrite INPUT.onnx OUTPUT.onnx [--report FILE]");
    auto in = fs::weakly_canonical(input), out = fs::weakly_canonical(output);
    if (in == out)
      throw std::runtime_error("Input and output must be different files");
    for (auto &auxiliary : {report, dump})
      if (!auxiliary.empty()) {
        auto p = fs::weakly_canonical(auxiliary);
        if (p == in || p == out)
          throw std::runtime_error(
              "Report/dump must not overwrite model files");
      }
    if (!report.empty() && !dump.empty() &&
        fs::weakly_canonical(report) == fs::weakly_canonical(dump))
      throw std::runtime_error("Report and dump must be different files");
    if (sdkJson) {
      std::ifstream file(input);
      if (!file)
        throw std::runtime_error("Cannot read SDK JSON input");
      std::string text((std::istreambuf_iterator<char>(file)), {}), ir;
      auto stats =
          vbx::rewriteSdkJson(text, options, dump.empty() ? nullptr : &ir);
      writeTextAtomic(out, text);
      std::ostringstream json;
      json << "{\n  \"engine\": \"MLIR 20.1.8\",\n  \"format\": "
              "\"tflite-sdk-json\",\n  \"nodes\": "
           << stats.nodes << ",\n  \"candidates\": " << stats.candidates
           << ",\n  \"rewrites\": " << stats.rewrites
           << ",\n  \"total_ms\": " << ms(totalStart) << ",\n  \"skipped\": {";
      bool firstSkip = true;
      for (const auto &[reason, count] : stats.skipped) {
        if (!firstSkip)
          json << ", ";
        firstSkip = false;
        json << '"' << reason << "\": " << count;
      }
      json << "}\n}\n";
      if (!dump.empty())
        writeText(dump, ir);
      if (!report.empty())
        writeText(report, json.str());
      std::cout << json.str();
      return 0;
    }
    auto start = Clock::now();
    std::ifstream file(input, std::ios::binary);
    onnx::ModelProto model;
    if (!file || !model.ParseFromIstream(&file))
      throw std::runtime_error("Cannot parse ONNX input");
    double parse = ms(start);
    if (in.parent_path() != out.parent_path() && vbx::hasExternalData(model))
      throw std::runtime_error("External tensor data requires output in the "
                               "input directory; data files are not copied");
    std::string ir;
    auto stats =
        vbx::rewriteModel(model, options, dump.empty() ? nullptr : &ir);
    start = Clock::now();
    writeModel(out, model);
    double serialize = ms(start);
    std::ostringstream json;
    json << std::setprecision(9)
         << "{\n  \"engine\": \"MLIR 20.1.8\",\n  \"input\": " << quoted(input)
         << ",\n  \"output\": " << quoted(output)
         << ",\n  \"nodes\": " << stats.nodes
         << ",\n  \"candidates\": " << stats.candidates
         << ",\n  \"attempts\": " << stats.attempts
         << ",\n  \"rewrites\": " << stats.rewrites
         << ",\n  \"timing_ms\": {\"parse\": " << parse
         << ", \"mlir_setup\": " << stats.setup_ms
         << ", \"import\": " << stats.import_ms
         << ", \"rewrite\": " << stats.rewrite_ms
         << ", \"export\": " << stats.export_ms
         << ", \"serialize\": " << serialize
         << ", \"total\": " << ms(totalStart) << "},\n  \"skipped\": {";
    bool first = true;
    for (auto &entry : stats.skipped) {
      if (!first)
        json << ", ";
      first = false;
      json << quoted(entry.first) << ": " << entry.second;
    }
    json << "}\n}\n";
    if (!dump.empty())
      writeText(dump, ir);
    if (!report.empty())
      writeText(report, json.str());
    std::cout << json.str();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "vbx-rewrite: " << e.what() << '\n';
    return 1;
  }
}
