# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.15)
# 添加torchconfig.cmake搜索路径, print(torch.utils.cmake_prefix_path)的输出结果
list(APPEND CMAKE_PREFIX_PATH "/opt/conda/lib/python3.11/site-packages/torch/share/cmake")
# 项目信息
project (Demo3 LANGUAGES CUDA CXX)
# set(CMAKE_BUILD_TYPE Debug)  # 启用调试模式
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# 添加 NVCC 编译选项，启用调试模式，便于使用nsight调试核函数
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
add_subdirectory(Add)
target_link_libraries(Add "${TORCH_LIBRARIES}")
