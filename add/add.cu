#include <torch/torch.h>
#include <cuda_runtime.h>

#include <math.h>

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

// 朴素版本
// __global__ void _addKernel(float* __restrict__  src1, float* __restrict__  src2, float* __restrict__ dst, int numel){
//     int m_idx = blockIdx.x*blockDim.x+threadIdx.x;
//     #pragma unroll
//     for(int idx = m_idx; idx < numel; idx += gridDim.x*blockDim.x){
//         dst[idx] = src1[idx] + src2[idx];
//     }
// }

// 使用float4 优化
__global__ void _addKernel(float* __restrict__  src1, float* __restrict__  src2, float* __restrict__ dst, int numel){
    int m_idx = blockIdx.x*blockDim.x+threadIdx.x;
	for(int i = m_idx; 4*i < numel; i += blockDim.x * gridDim.x) {
		float4 a = reinterpret_cast<float4*>(src1)[i];
		float4 b = reinterpret_cast<float4*>(src2)[i];
		float4 c;
		c.x = a.x + b.x;
		c.y = a.y + b.y;
		c.z = a.z + b.z;
		c.w = a.w + b.w;
		reinterpret_cast<float4*>(dst)[i] = c;
    }
}
void add(torch::Tensor src1_tensor, torch::Tensor src2_tensor, torch::Tensor dst_tensor, int numel){
    // 确保输入和输出张量都是在CUDA上
    TORCH_CHECK(src1_tensor.is_cuda(), "src1 tensor must be on the GPU");
    TORCH_CHECK(src2_tensor.is_cuda(), "src2 tensor must be on the GPU");
    TORCH_CHECK(dst_tensor.is_cuda(), "dst tensor must be on the GPU");
    float *src1 = src1_tensor.data_ptr<float>();
    float *src2 = src2_tensor.data_ptr<float>();
    float *dst = dst_tensor.data_ptr<float>();
    int block_x = 1024;
    //如果使用float4类型，一个线程处理四个元素，所以算grid时要将block_x*4
    int grid_x = (numel + 4*block_x - 1) / (4*block_x);
    // grid_x = grid_x>65536 ? 65536 : grid_x;
    dim3 grid_dim(grid_x, 1);
    dim3 block_dim(block_x, 1);

    _addKernel<<<grid_dim, block_dim>>>(src1, src2, dst, numel);
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());
}