#include <torch/extension.h>

// 声明函数
void _sum(torch::Tensor src_tensor, torch::Tensor dst_tensor, int numel, int method = 0);
void _max(torch::Tensor src_tensor, torch::Tensor dst_tensor, int numel, int method = 0);
void _min(torch::Tensor src_tensor, torch::Tensor dst_tensor, int numel, int method = 0);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      // 第一个参数表示注册到python模块中的函数名称，可以替换为其他名字，使用方法为：模块.add_cuda
      // 第二个参数是上面编写的kernel launch 函数，这里需要获得该函数的地址
      // 第三个参数是函数描述，可以修改
      // 后面的py::arg是用来为add_cuda定义参数的，这些参数的数目，顺序必须和add保持一致
      m.def("sum", &_sum, "Cuda Core reduce function", py::arg("src"), py::arg("dst"), py::arg("size"), py::arg("native"));
      m.def("max", &_max, "Cuda Core reduce function", py::arg("src"), py::arg("dst"), py::arg("size"), py::arg("native"));
      m.def("min", &_min, "Cuda Core reduce function", py::arg("src"), py::arg("dst"), py::arg("size"), py::arg("native"));
}