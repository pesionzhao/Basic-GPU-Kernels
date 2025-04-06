#include "softmax.cuh"
#include <torch/torch.h>

#define cudaCheckError(ans)                   \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename KernelFunc>
void launchSoftMaxKernel(
    torch::Tensor src_tensor,
    torch::Tensor dst_tensor,
    int M,
    int N,
    KernelFunc kernel       // 函数作为模板参数传入
) {
    TORCH_CHECK(src_tensor.is_cuda(), "src tensor must be on the GPU");
    TORCH_CHECK(dst_tensor.is_cuda(), "dst tensor must be on the GPU");

    const float* src = src_tensor.data_ptr<float>();
    float* dst = dst_tensor.data_ptr<float>();
    kernel(src, dst, M, N);

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());
}

// 只有N<=1024的情况使用
void launchSoftMax2dwarp(const float* src, float* dst, int M, int N){
    const int block_size_x = 32; // block_size_x必须等于32
    const int block_size_y = 4;
    dim3 blocks(block_size_x, block_size_y);
    int grid_size = (M+block_size_y-1)/block_size_y;
    dim3 grids(grid_size);
    if (N<0) return;
    #define DEFINE_ONE_ELIF(col_per_thread) \
    else if(N<=block_size_x*col_per_thread){ \
        softmax2d_warp<col_per_thread, block_size_x><<<grids, blocks>>>(src, dst, M, N); \
    }
    DEFINE_ONE_ELIF(1)
    DEFINE_ONE_ELIF(2)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(8)
    DEFINE_ONE_ELIF(16)
    DEFINE_ONE_ELIF(32)
    #undef DEFINE_ONE_ELIF
}

// N>1024的情况使用, 一个block处理一行
void launchSoftMax2dShared(const float* src, float* dst, int M, int N){
    const int block_size_x = 128;
    softmax2d_shared<block_size_x><<<M, block_size_x, N*sizeof(float)>>>(src, dst, M, N);
}

void softmax(torch::Tensor src_tensor, torch::Tensor dst_tensor, int M, int N){
    if(N<1024)
        launchSoftMaxKernel<decltype(launchSoftMax2dwarp)>(src_tensor, dst_tensor, M, N, launchSoftMax2dwarp);
    else
        launchSoftMaxKernel<decltype(launchSoftMax2dShared)>(src_tensor, dst_tensor, M, N, launchSoftMax2dShared);
}