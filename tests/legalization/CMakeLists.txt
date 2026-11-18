add_executable(vbx-rewrite-tests rewrite_test.cpp)
target_link_libraries(vbx-rewrite-tests PRIVATE vbx_rewrite)
add_test(NAME vbx-rewrite-core COMMAND vbx-rewrite-tests)
