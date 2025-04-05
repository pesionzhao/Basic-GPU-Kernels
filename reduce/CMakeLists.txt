# 用于调试cuda程序
add_library(static_lib STATIC reduce.cu)
target_link_libraries(static_lib PUBLIC "${TORCH_LIBRARIES}")
add_executable(reduce check.cu)
target_link_libraries(reduce PRIVATE static_lib)
# add_executable(Reduce reduce.cu check.cu)