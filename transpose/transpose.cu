#include<torch/extension.h>

// 朴素算子
__global__ void  _transposeKernel_native(float* __restrict__ src, float* __restrict__ dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
        float tmp = src[row*N + col];
        dst[col*M + row] = tmp;
    }
}
//利用共享内存合并写入合并访问
template<int BLOCK_SIZE>
__global__ void  _transposeKernel_v1(float* __restrict__ src, float* __restrict__ dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float cache[BLOCK_SIZE][BLOCK_SIZE];
    if(row < M && col < N){
        //写入的时候就直接转置好，但这里也会引起bank冲突，经测试，读取时避免bank冲突比写入时避免bank冲突性能高
        cache[threadIdx.x][threadIdx.y] = src[row*N+col];
    }
    __syncthreads();
    //块首地址+块内索引
    int dst_row = blockIdx.x * blockDim.x + threadIdx.y;
    int dst_col = blockIdx.y * blockDim.y + threadIdx.x;
    if(dst_row < N && dst_col < M){
        //读取无bank冲突
        dst[dst_row*M+dst_col] = cache[threadIdx.y][threadIdx.x];
    }
}
//避免bank冲突的方法
template<int BLOCK_SIZE>
__global__ void  _transposeKernel_v2(float* __restrict__ src, float* __restrict__ dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //防止bank冲突，更改cache的大小，使得相邻threadIdx.x相差33，而不是32(同一个bank)
    __shared__ float cache[BLOCK_SIZE][BLOCK_SIZE+1];
    if(row < M && col < N){
        cache[threadIdx.y][threadIdx.x] = src[row*N+col];
    }
    __syncthreads();
    //块首地址+块内索引
    int dst_row = blockIdx.x * blockDim.x + threadIdx.y;
    int dst_col = blockIdx.y * blockDim.y + threadIdx.x;
    if(dst_row < N && dst_col < M){
        dst[dst_row*M+dst_col] = cache[threadIdx.x][threadIdx.y];
    }
}

void transpose(torch::Tensor src_tensor, torch::Tensor dst_tensor, int M, int N, int method=0){
    const int block_size = 32;
    dim3 block(block_size, block_size);
    dim3 grid((N+block_size-1)/block_size, (M+block_size-1)/block_size);
    switch(method){
        // 使用共享内存合并读写全局内存
        case 1:
            _transposeKernel_v1<block_size><<<grid, block>>>(src_tensor.data_ptr<float>(), dst_tensor.data_ptr<float>(), M, N);
            break;
        // 避免bank冲突后的版本
        case 2:
            _transposeKernel_v2<block_size><<<grid, block>>>(src_tensor.data_ptr<float>(), dst_tensor.data_ptr<float>(), M, N);
            break;
        // 朴素版本
        default:
            _transposeKernel_native<<<grid, block>>>(src_tensor.data_ptr<float>(), dst_tensor.data_ptr<float>(), M, N);
            break;
    }
}