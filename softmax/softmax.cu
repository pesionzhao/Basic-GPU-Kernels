#include <math_constants.h>
#include <stdio.h>
#include <cub/block/block_reduce.cuh>
struct SumOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const { return a + b; }
};

struct MaxOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const { return max(a,b); }
};

template<class ReductionOp, int thread_group>
__device__ float warpReuce(float val){
    for(int offset = thread_group>>1; offset>0; offset>>=1){
        val = ReductionOp()(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

//blocksize>32时， 借助cub进行归约, 也可以自己实现，先线程束归约，再把归约结果放到一个线程束里
template<class ReductionOp, int BLOCK_SIZE>
__inline__ __device__ float blockReduce(float val){
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float result_broadcast;
    float result = BlockReduce(temp_storage).Reduce(val, ReductionOp());
    if (threadIdx.x == 0) { result_broadcast = result; }
    __syncthreads();
    return result_broadcast;
}

// 一个一维block
template<int cols_per_thread, int BLOCK_SIZE>
__global__ void softmax_native(float* src, float* dst, int N){
    constexpr int num_packs = (cols_per_thread+3) / 4;
    float local_max = -CUDART_INF_F;
    float4 buf[num_packs];//寄存器存num_packs*4个变量
    for(int i = threadIdx.x; i<N; i+=blockDim.x){
        local_max = max(local_max, src[i]);
    }
    //此时归约到了一个block中
    local_max = blockReduce<MaxOp, BLOCK_SIZE>(local_max);
    float local_sum = 0.0f;
    for(int i = threadIdx.x; i<N; i+=blockDim.x){
        dst[i] = exp(src[i]-local_max);
        local_sum += dst[i];
    }
    local_sum = blockReduce<SumOp, BLOCK_SIZE>(local_sum);
    for(int i = threadIdx.x; i<N; i+=blockDim.x){
        dst[i] /= local_sum;
    }
}

// 一个一维block 优化版
template<int cols_per_thread, int BLOCK_SIZE>
__global__ void softmax(float* src, float* dst, int N){
    constexpr int num_packs = (cols_per_thread+3) / 4;
    float local_max = -CUDART_INF_F;
    float4 buf[num_packs];//寄存器存num_packs*4个变量
    for(int i = 0; i<num_packs; i++){
        int col = (i*BLOCK_SIZE+threadIdx.x)*4;
        if(col<N){
            buf[i] = {src[col], src[col+1], src[col+2], src[col+3]};
        }
        else{
            buf[i].x = -CUDART_INF_F;
            buf[i].y = -CUDART_INF_F;
            buf[i].z = -CUDART_INF_F;
            buf[i].x = -CUDART_INF_F;
        }
        local_max = max(local_max, max(max(buf[i].x, buf[i].y), max(buf[i].z, buf[i].w)));
    }
    //此时归约到了一个block中
    local_max = blockReduce<MaxOp, BLOCK_SIZE>(local_max);
    float local_sum = 0.0f;
    for(int i = 0; i<num_packs; i++){
        buf[i].x = exp(buf[i].x - local_max);
        buf[i].y = exp(buf[i].y - local_max);
        buf[i].z = exp(buf[i].z - local_max);
        buf[i].w = exp(buf[i].w - local_max);
        local_sum += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
    }
    local_sum = blockReduce<SumOp, BLOCK_SIZE>(local_sum);
    for(int i = 0; i<num_packs; i++){
        int col = (i*BLOCK_SIZE+threadIdx.x)*4;
        if(col<N){
            dst[col] =  buf[i].x/local_max;
            dst[col+1] =  buf[i].y/local_max;
            dst[col+2] =  buf[i].z/local_max;
            dst[col+3] =  buf[i].w/local_max;
        }
    }
}