add_subdirectory(Legalization)
# Preserve the library entry point while keeping implementation ownership clear.
add_library(vbx_rewrite INTERFACE)
target_link_libraries(vbx_rewrite INTERFACE vbx_onnx_frontend)
add_subdirectory(Tools)
