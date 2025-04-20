#include<stdio.h>
#include<assert.h>
#include <exception>
#include <iostream>
#define Br 8
#define Bc 8
#define RED   "\033[1;31m"
#define RESET "\033[0m"
#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }

// 专门用于 kernel 调用之后的检查
#define CHECK_KERNEL()                                      \
    {                                                       \
        CUDA_CHECK(cudaGetLastError());                     \
        CUDA_CHECK(cudaDeviceSynchronize());                \
    }
// 错误检查实现函数
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << RED << "[CUDA ERROR] " << cudaGetErrorString(code)
                  << " at " << file << ":" << line << RESET << std::endl;
        if (abort) exit(code);
    }
}

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

template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flashAttentionGEMMV1Kernel(const float* q, const float* k, const float* v, float* dst, int batch, int num_head, int seq_len, int head_dim){
    //假设Q/K/V shape [batch, num_head, seq_len, head_dim]
    //block [block_size, head_dim], grid[M/block_size, bc*num_head]
    int bid = blockIdx.x*BLOCK_SIZE;
    int row = bid+threadIdx.y;
    int col = threadIdx.x;
    __shared__ float local_O[Br][Bc];
    int num_pack = (seq_len + Bc - 1)/Bc;
    const float* intputK = k+blockIdx.y*seq_len*head_dim;
    const float* intputV = v+blockIdx.y*seq_len*head_dim;
    __shared__ float intputQ[Br*HEAD_DIM];
    intputQ[threadIdx.y*HEAD_DIM+col] = q[blockIdx.y*seq_len*head_dim+row*HEAD_DIM+col];//固定Q
    __syncthreads();
    for(int pack_id = 0; pack_id<num_pack; pack_id++){
        float qk_res = 0.0f;
        for(int i = 0; i<head_dim; i++){
            // qk_res += intputQ[threadIdx.y*HEAD_DIM+i]*intputK[(col%Bc+pack_id*Bc)*HEAD_DIM+i];// Q*K^T
            qk_res += intputQ[threadIdx.y*HEAD_DIM+i]*intputK[col%Bc+pack_id*Bc+seq_len*i];// Q * K
        }
        local_O[threadIdx.y][col%Bc]=qk_res;
        intputQ[threadIdx.y*HEAD_DIM+col]=0.0f;
        __syncthreads();
        for(int i = 0; i<Bc;i++){
            intputQ[threadIdx.y*HEAD_DIM+col]+=local_O[threadIdx.y][i]*intputV[(i+pack_id*Bc)*HEAD_DIM+col];
        }
        __syncthreads();
    }
    dst[blockIdx.y*seq_len*head_dim+row*HEAD_DIM+col] = intputQ[threadIdx.y*HEAD_DIM+col];
};

template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flashAttentionV1Kernel(const float* q, const float* k, const float* v, float* dst, int batch, int num_head, int seq_len, int head_dim){
    //假设Q/K/V shape [batch, num_head, seq_len, head_dim]
    //block [block_size, head_dim], grid[M/block_size, bc*num_head]
    int bid = blockIdx.x*BLOCK_SIZE;
    int row = bid+threadIdx.y;
    int col = threadIdx.x;
    __shared__ float local_O[Br][Bc];
    int num_pack = (seq_len + Bc - 1)/Bc;
    float old_max = 0;
    float old_d = 0;
    const float* intputK = k+blockIdx.y*seq_len*head_dim;
    const float* intputV = v+blockIdx.y*seq_len*head_dim;
    __shared__ float intputQ[Br*HEAD_DIM];
    __shared__ float outputO[Br*HEAD_DIM];
    outputO[threadIdx.y*HEAD_DIM+col] = 0.0f;
    intputQ[threadIdx.y*HEAD_DIM+col] = q[blockIdx.y*seq_len*head_dim+row*HEAD_DIM+col];//固定Q
    __syncthreads();
    for(int pack_id = 0; pack_id<num_pack; pack_id++){
        float qk_res = 0.0f;
        for(int i = 0; i<head_dim; i++){
            qk_res += intputQ[threadIdx.y*HEAD_DIM+i]*intputK[(pack_id*Bc+col%Bc)*HEAD_DIM+i];// Q*K^T
            // qk_res += intputQ[threadIdx.y*HEAD_DIM+i]*intputK[col%Bc+pack_id*Bc+seq_len*i];// Q * K
        }
        //由于x方向线程数为head_dim, 大于bc, 在进行乘法时只需要bc个线程参与计算即可，这里使用求余防止分支发散，x方向上，每个bc块的元素都一样
        float cur_max = qk_res;
        // online softmax
        cur_max = warpReduce<MaxOp, Bc>(qk_res);
        cur_max = max(cur_max, old_max);
        float cur_sum_tmp = __expf(qk_res-cur_max);
        local_O[threadIdx.y][col%Bc]=cur_sum_tmp; // 不用同步，因为每个线程操作对应的shared memory
        float cur_sum = warpReduce<SumOp, Bc>(cur_sum_tmp); //只有Br行Bc的线程有用
        float alpha = __expf(old_max-cur_max);
        float new_d = alpha*old_d+cur_sum;
        local_O[threadIdx.y][col%Bc]/=new_d;
        outputO[threadIdx.y*HEAD_DIM+col] = outputO[threadIdx.y*HEAD_DIM+col]*old_d/new_d*alpha;
        __syncthreads();
        for(int i = 0; i<Bc;i++){
            outputO[threadIdx.y*HEAD_DIM+col]+=local_O[threadIdx.y][i]*intputV[(i+pack_id*Bc)*HEAD_DIM+col];
        }
        old_max = cur_max;
        old_d = new_d;
    }
    dst[blockIdx.y*seq_len*head_dim+row*HEAD_DIM+col] = outputO[threadIdx.y*HEAD_DIM+col];
};

template<int BLOCK_SIZE>
__global__ void GEMM(const float* src1, const float* src2, const float* dst, int M, int N, int K){
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int tid = threadIdx.y*blockDim.y+threadIdx.x;
    __shared__ float sa[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float sb[BLOCK_SIZE*BLOCK_SIZE];
    int num_tile = (K+BLOCK_SIZE-1)/BLOCK_SIZE;
    float tmp = 0;
    for(int i = 0; i<num_tile; i++){
        sa[tid] = src1[row*K+i*BLOCK_SIZE+threadIdx.x];
        sa[tid] = src1[(i*BLOCK_SIZE+threadIdx.y)*N+col];
        __syncthreads();
        for(int k = 0; k<BLOCK_SIZE; k++){
            tmp += sa[threadIdx.y*BLOCK_SIZE+k]*sb[k*BLOCK_SIZE+threadIdx.x];
        }
    }
    dst[row*N+col] = tmp;
}

template<int HEAD_DIM>
void lauchFA(const float* q, const float* k, const float* v, float* dst, int batch, int num_head, int seq_len, int head_dim=64){
    // assert(head_dim==16);
    int grid_size_y = batch*num_head;
    int grid_size_x = (seq_len+Br-1)/Br;
    dim3 block(HEAD_DIM, Br);
    flashAttentionV1Kernel<Br, HEAD_DIM><<<grid_size_x, block>>>(q,k,v,dst,batch,num_head,seq_len,head_dim);
    CHECK_KERNEL();
}