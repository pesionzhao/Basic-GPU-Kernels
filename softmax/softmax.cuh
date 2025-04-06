#include <math_constants.h>
#include <stdio.h>
#include <cub/block/block_reduce.cuh>
#include <cassert>
struct SumOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const { return a + b; }
};

struct MaxOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const { return max(a,b); }
};

template<class ReductionOp, int thread_group>
__device__ float warpReduce(float val){
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
__global__ void softmax1d(const float* src, float* dst, int N){
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
    __syncthreads();
    float local_sum = 0.0f;
    for(int i = 0; i<num_packs; i++){
        buf[i].x = exp(buf[i].x - local_max);
        buf[i].y = exp(buf[i].y - local_max);
        buf[i].z = exp(buf[i].z - local_max);
        buf[i].w = exp(buf[i].w - local_max);
        local_sum += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
    }
    local_sum = blockReduce<SumOp, BLOCK_SIZE>(local_sum);
    __syncthreads();
    for(int i = 0; i<num_packs; i++){
        int col = (i*BLOCK_SIZE+threadIdx.x)*4;
        if(col<N){
            dst[col] =  buf[i].x/local_sum;
            dst[col+1] =  buf[i].y/local_sum;
            dst[col+2] =  buf[i].z/local_sum;
            dst[col+3] =  buf[i].w/local_sum;
        }
    }
}

// 二维softmax一个block处理一行数据
// block  <BLOCK_SIZE>, grid <M>
template<int cols_per_thread, int BLOCK_SIZE> // cols_per_thread决定了每个寄存器要存多少变量
__global__ void softmax2d_block(const float* src, float* dst, int M, int N){
    assert(blockDim.y==1);
    constexpr int num_packs = (cols_per_thread+3) / 4;
    int row_idx = blockIdx.x; //因为blockDim.y==1， threadIdx.y==0
    float4 buf[num_packs];//寄存器存num_packs*4个变量
    for(int row=row_idx; row<M; row+=gridDim.x){
        float local_max = -CUDART_INF_F;
        const float* inputptr = src+row*(N);
        float* outputptr = dst+row*(N);
        for(int i = 0; i<num_packs; i++){
            int col = (i*BLOCK_SIZE+threadIdx.x)*4;
            if(col<N){
                buf[i] = {inputptr[col], inputptr[col+1], inputptr[col+2], inputptr[col+3]};
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
        local_max = warpReduce<MaxOp, 32>(local_max);
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
                outputptr[col] =  buf[i].x/local_sum;
                outputptr[col+1] =  buf[i].y/local_sum;
                outputptr[col+2] =  buf[i].z/local_sum;
                outputptr[col+3] =  buf[i].w/local_sum;
            }
        }
    }
}

// 二维softmax 二维block
// block  <32， 4>, grid <M/4>
template<int cols_per_thread, int BLOCK_SIZE> // cols_per_thread决定了每个寄存器要存多少变量
__global__ void softmax2d_warp(const float* src, float* dst, int M, int N){
    static_assert(BLOCK_SIZE == 32, "warp reduce block size must equal to 32");
    constexpr int num_packs = (cols_per_thread+3) / 4; //用几个float4打包, 一个线程操作的float4元素个数
    int row_idx = blockDim.y*blockIdx.x + threadIdx.y;
    float4 buf[num_packs];//寄存器存num_packs*4个变量
    for(int row=row_idx; row<M; row+=gridDim.x){
        float local_max = -CUDART_INF_F;
        const float* inputptr = src+row*(N);
        float* outputptr = dst+row*(N);
        for(int i = 0; i<num_packs; i++){
            int col = (i*BLOCK_SIZE+threadIdx.x)*4;
            if(col<N){
                buf[i] = {inputptr[col], inputptr[col+1], inputptr[col+2], inputptr[col+3]};
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
        local_max = warpReduce<MaxOp, BLOCK_SIZE>(local_max);
        float local_sum = 0.0f;
        for(int i = 0; i<num_packs; i++){
            buf[i].x = exp(buf[i].x - local_max);
            buf[i].y = exp(buf[i].y - local_max);
            buf[i].z = exp(buf[i].z - local_max);
            buf[i].w = exp(buf[i].w - local_max);
            local_sum += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
        }
        local_sum = warpReduce<SumOp, BLOCK_SIZE>(local_sum);
        for(int i = 0; i<num_packs; i++){
            int col = (i*BLOCK_SIZE+threadIdx.x)*4;
            if(col<N){
                outputptr[col] =  buf[i].x/local_sum;
                outputptr[col+1] =  buf[i].y/local_sum;
                outputptr[col+2] =  buf[i].z/local_sum;
                outputptr[col+3] =  buf[i].w/local_sum;
            }
        }
    }
}

// 二维softmax 一维block
// block  <BLOCK_SIZE>, grid <M>
template</*int cols_per_thread,*/ int BLOCK_SIZE> // 由于使用共享内存，所以不用关心每个线程寄存器存多少值，只关心共享内存一共存多少值，也就是N
__global__ void softmax2d_shared(const float* src, float* dst, int M, int N){
    assert(blockDim.y==1);
    int num_packs = N>>2; 
    int row_idx = blockIdx.x;
    // __shared__ float buf[BLOCK_SIZE*cols_per_thread];
    extern __shared__ float buf[]; //外部传递 N 
    for(int row=row_idx; row<M; row+=gridDim.x){
        float local_max = -CUDART_INF_F;
        const float* inputptr = src+row*(N);
        float* outputptr = dst+row*(N);
        for(int pack_id = threadIdx.x; pack_id < num_packs; pack_id += BLOCK_SIZE){
            int col = pack_id*4;
            float4 tmp = {inputptr[col], inputptr[col+1], inputptr[col+2], inputptr[col+3]};
            buf[pack_id] = tmp.x;
            buf[pack_id+num_packs] = tmp.y;
            buf[pack_id+num_packs*2] = tmp.z;
            buf[pack_id+num_packs*3] = tmp.w;
            local_max = max(local_max, max(max(tmp.x, tmp.y), max(tmp.z, tmp.w)));
        }
        //此时归约到了一个block中
        local_max = blockReduce<MaxOp, BLOCK_SIZE>(local_max);
        float local_sum = 0.0f;
        for(int i = threadIdx.x; i < N; i += BLOCK_SIZE){
            buf[i] = exp(buf[i] - local_max);
            local_sum += buf[i];
        }
        local_sum = blockReduce<SumOp, BLOCK_SIZE>(local_sum);
        for(int pack_id = threadIdx.x; pack_id < num_packs; pack_id += BLOCK_SIZE){
            int col = 4*pack_id;
            outputptr[col] = buf[pack_id]/local_sum;
            outputptr[col+1] = buf[pack_id+num_packs]/local_sum;
            outputptr[col+2] = buf[pack_id+num_packs*2]/local_sum;
            outputptr[col+3] = buf[pack_id+num_packs*3]/local_sum;
        }
    }
}