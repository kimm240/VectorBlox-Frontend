add_subdirectory(Frontend/ONNX)
add_subdirectory(Target/VectorBlox)
# Preserve the library entry point while keeping implementation ownership clear.
add_library(vbx_rewrite INTERFACE)
target_link_libraries(vbx_rewrite INTERFACE vbx_onnx_frontend vbx_vectorblox_target)
add_subdirectory(Tools)
