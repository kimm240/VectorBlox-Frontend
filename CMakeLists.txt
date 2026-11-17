cmake_minimum_required(VERSION 3.20)
project(vbx_onnx_rewrite VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# All third-party sources/build products live inside this checkout.
# No installed MLIR, LLVM or Protobuf is discovered or used.
set(VBX_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/.deps/src")
set(VBX_LLVM "${VBX_DEPS}/llvm-project")
set(VBX_PROTOBUF "${VBX_DEPS}/protobuf-3.20.3")
if(NOT EXISTS "${VBX_LLVM}/llvm/CMakeLists.txt" OR NOT EXISTS "${VBX_PROTOBUF}/cmake/CMakeLists.txt")
  message(FATAL_ERROR "Run: python3 tools/bootstrap.py --fetch-only")
endif()
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "" FORCE)
foreach(flag LLVM_INCLUDE_TESTS LLVM_INCLUDE_BENCHMARKS LLVM_INCLUDE_EXAMPLES LLVM_BUILD_TOOLS
  LLVM_ENABLE_ASSERTIONS LLVM_ENABLE_ZLIB LLVM_ENABLE_ZSTD LLVM_ENABLE_LIBXML2 LLVM_ENABLE_LIBEDIT
  LLVM_ENABLE_CURL LLVM_ENABLE_FFI LLVM_ENABLE_TERMINFO MLIR_INCLUDE_TESTS MLIR_INCLUDE_EXAMPLES
  MLIR_ENABLE_BINDINGS_PYTHON MLIR_ENABLE_EXECUTION_ENGINE LLVM_INCLUDE_UTILS LLVM_INCLUDE_DOCS MLIR_INCLUDE_DOCS
  MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  set(${flag} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
add_subdirectory("${VBX_LLVM}/llvm" "deps/llvm" EXCLUDE_FROM_ALL)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
add_subdirectory("${VBX_PROTOBUF}/cmake" "deps/protobuf" EXCLUDE_FROM_ALL)
# Generate the full-runtime schema for recursive external-data inspection.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/onnx.proto" VBX_ONNX_PROTO)
string(REPLACE "option optimize_for = LITE_RUNTIME;" "option optimize_for = SPEED;" VBX_ONNX_PROTO "${VBX_ONNX_PROTO}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/schema")
# Preserve the timestamp when schema contents are unchanged, so reconfiguring
# the project does not regenerate protobuf headers and rebuild every pass.
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schema/onnx.proto"
  CONTENT "${VBX_ONNX_PROTO}" @ONLY)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc" "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h"
  COMMAND $<TARGET_FILE:protoc> --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
          -I${CMAKE_CURRENT_BINARY_DIR}/schema
          ${CMAKE_CURRENT_BINARY_DIR}/schema/onnx.proto
  DEPENDS protoc third_party/onnx/onnx.proto "${CMAKE_CURRENT_BINARY_DIR}/schema/onnx.proto" VERBATIM)
add_library(vbx_onnx_schema STATIC "${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc")
target_include_directories(vbx_onnx_schema PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(vbx_onnx_schema PUBLIC libprotobuf)
# Shared include/link requirements for the two compiler components.
add_library(vbx_compiler_dependencies INTERFACE)
target_include_directories(vbx_compiler_dependencies INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(vbx_compiler_dependencies SYSTEM INTERFACE
  "${VBX_LLVM}/llvm/include" "${CMAKE_CURRENT_BINARY_DIR}/deps/llvm/include"
  "${VBX_LLVM}/mlir/include" "${CMAKE_CURRENT_BINARY_DIR}/deps/llvm/tools/mlir/include")
target_link_libraries(vbx_compiler_dependencies INTERFACE
  vbx_onnx_schema MLIRIR MLIRFuncDialect MLIRPass MLIRTransformUtils MLIRTransforms)
add_subdirectory(src)
option(VBX_ENABLE_LTO "Enable LTO for project sources (not entire LLVM)" ON)
if(VBX_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT VBX_IPO_SUPPORTED OUTPUT VBX_IPO_ERROR)
  if(VBX_IPO_SUPPORTED)
    set_property(TARGET vbx_onnx_frontend vbx_vectorblox_target vbx-rewrite
      PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()
option(VBX_BUILD_TESTS "Build separate regression tests for development/CI" OFF)
if(VBX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
