add_library(vbx_onnx_frontend STATIC ImportExport.cpp Passes.cpp ChannelMean.cpp Legalize.cpp Unfold.cpp ConvBranches.cpp)
target_link_libraries(vbx_onnx_frontend PUBLIC vbx_compiler_dependencies)
target_compile_options(vbx_onnx_frontend PRIVATE -Wall -Wextra -Wpedantic)
