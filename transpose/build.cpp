#include <torch/extension.h>

// 声明函数,method代表矩阵转置的不同实现
void transpose(torch::Tensor src1_tensor, torch::Tensor dst_tensor, int M, int N, int method=0);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("transpose_cuda", &transpose, "Cuda Core softmax function",
            py::arg("src1"), py::arg("dst"), py::arg("M"), py::arg("N"), py::arg("method"));
}