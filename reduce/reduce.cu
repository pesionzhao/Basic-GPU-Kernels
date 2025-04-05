#include <math_constants.h>
// 参考自oneflow https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh
// 定义了模板函数 Inf()，并对不同数据类型进行特化
#include <cub/block/block_reduce.cuh>
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

template <typename T>
__inline__ __device__ T Inf();

//模板特化
template <>
__inline__ __device__ float Inf<float>() {
return CUDART_INF_F;
}
template<>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

//__forceinline__代表强制内联
template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
  __device__ __forceinline__ T init(){return 0;}
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a,b); }
  __device__ __forceinline__ T init() const {return -Inf<T>();}
};

template<typename T>
struct MinOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return min(a,b); }
  __device__ __forceinline__ T init() const {return Inf<T>();}
};

//束内归约thread_group_width最大为32
//ReductionOp规定求和还是求最大值
template<template<typename> class ReductionOp, typename T, int thread_group_width>
__inline__ __device__ T warpReduce(T val){
    for(int offset = thread_group_width>>1; offset>0; offset>>=1){
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

//blocksize>32时， 借助cub进行归约, 也可以自己实现，先线程束归约，再把归约结果放到一个线程束里
template<template<typename> class ReductionOp, typename T, int BLOCK_SIZE>
__inline__ __device__ T blockReduce(T val){
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
    if (threadIdx.x == 0) { result_broadcast = result; }
    __syncthreads();
    return result_broadcast;
}

//一维向量归约
template<template<typename> class ReductionOp, typename T, int BLOCK_SIZE>
__device__ void reduce_native(const T* src, T* dst, int N){
    constexpr int num_warp_per_block = BLOCK_SIZE/32;
    int tid = threadIdx.x;
    __shared__ T sum[BLOCK_SIZE/32];//BLOCKSIZE/32一定小于等于32，所以一定可以用线程束归约
    T res = ReductionOp<T>().init();
    //首先归约到一个block中
    for(int i = threadIdx.x; i<N; i+=blockDim.x){
        res = ReductionOp<T>()(src[i], res);
    }
    // 等价于 res = blockReduce<ReductionOp, T, BLOCK_SIZE>(res);
    //相邻32个线程完成归约
    res = warpReduce<ReductionOp, T, 32>(res);
    //此时每个线程束所有线程的结果都为束内归约结果
    if(tid%32==0)
    sum[tid/32] = res;//转移到共享内存，为了之后放到同一个线程束的寄存器中
    __syncthreads();
    if(tid<num_warp_per_block){
        res = sum[tid];
        res = warpReduce<ReductionOp, T, num_warp_per_block>(res);
    }
    if(tid==0)
        *dst = res;
}

//一维向量归约, 借助cub
template<template<typename> class ReductionOp, typename T, int BLOCK_SIZE>
__device__ void reduce_cub(const T* src, T* dst, int N){
    int tid = threadIdx.x;
    __shared__ T sum[BLOCK_SIZE/32];//BLOCKSIZE/32一定小于等于32，所以一定可以用线程束归约
    T res = ReductionOp<T>().init();
    //首先归约到一个block中
    for(int i = threadIdx.x; i<N; i+=blockDim.x){
        res = ReductionOp<T>()(src[i], res);
    }
    res = blockReduce<ReductionOp, T, BLOCK_SIZE>(res);
    if(tid==0)
        *dst = res;
}


template<typename T, int BLOCK_SIZE>
__global__ void sumKernel(const T* src, T* dst, int N, int method=0){
    if(method)
        reduce_native<SumOp, T, BLOCK_SIZE>(src, dst, N);
    else
        reduce_cub<SumOp, T, BLOCK_SIZE>(src, dst, N);
}

template<typename T, int BLOCK_SIZE>
__global__ void maxKernel(const T* src, T* dst, int N, int method=0){
    if(method)
        reduce_native<MaxOp, T, BLOCK_SIZE>(src, dst, N);
    else
        reduce_cub<MaxOp, T, BLOCK_SIZE>(src, dst, N);
}

template<typename T, int BLOCK_SIZE>
__global__ void minKernel(const T* src, T* dst, int N, int method=0){
    if(method)
        reduce_native<MinOp, T, BLOCK_SIZE>(src, dst, N);
    else
        reduce_cub<MinOp, T, BLOCK_SIZE>(src, dst, N);
}

template <typename KernelFunc, typename T, int BLOCK_SIZE>
void launchReductionKernel(
    torch::Tensor src_tensor,
    torch::Tensor dst_tensor,
    int numel,
    int method,
    KernelFunc kernel       // 核函数作为模板参数传入
) {
    TORCH_CHECK(src_tensor.is_cuda(), "src tensor must be on the GPU");
    TORCH_CHECK(dst_tensor.is_cuda(), "dst tensor must be on the GPU");

    const T* src = src_tensor.data_ptr<T>();
    T* dst = dst_tensor.data_ptr<T>();

    dim3 block_dim(BLOCK_SIZE);
    kernel<<<1, block_dim>>>(src, dst, numel, method);

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());
}

void _sum(torch::Tensor src_tensor, torch::Tensor dst_tensor, int numel, int method = 0){
    launchReductionKernel<decltype(sumKernel<float, 64>), float, 64>(src_tensor, dst_tensor, numel, method, sumKernel<float, 64>);
}

void _max(torch::Tensor src_tensor, torch::Tensor dst_tensor, int numel, int method = 0){
    launchReductionKernel<decltype(maxKernel<float, 64>), float, 64>(src_tensor, dst_tensor, numel, method, maxKernel<float, 64>);
}

void _min(torch::Tensor src_tensor, torch::Tensor dst_tensor, int numel, int method = 0){
    launchReductionKernel<decltype(minKernel<float, 64>), float, 64>(src_tensor, dst_tensor, numel, method, minKernel<float, 64>);
}


template __global__ void sumKernel<float, 128>(float const*, float*, int, int);
