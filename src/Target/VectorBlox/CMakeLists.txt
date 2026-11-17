add_library(vbx_vectorblox_target STATIC Graph.cpp Layout.cpp Passes.cpp InverseRegroup.cpp CanonicalizeViews.cpp)
target_link_libraries(vbx_vectorblox_target PUBLIC vbx_compiler_dependencies)
target_compile_options(vbx_vectorblox_target PRIVATE -Wall -Wextra -Wpedantic)
