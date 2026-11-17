add_executable(vbx-rewrite vbx-rewrite.cpp)
target_link_libraries(vbx-rewrite PRIVATE vbx_rewrite)
target_compile_options(vbx-rewrite PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(vbx-rewrite PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
