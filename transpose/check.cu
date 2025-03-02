/***
check.cu用来调试cuda代码保证正确性，不参与python包的封装
***/ 

#include<stdio.h>
// 朴素算子, 尽量保证合并写入
__global__ void  _transposeKernel_native(float* __restrict__ src, float* __restrict__ dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < N){
        float tmp = src[col*M + row];
        dst[row*N + col] = tmp;
    }
}
//利用共享内存合并写入合并访问，v1/v2代表两种避免bank冲突的方法
template<int BLOCK_SIZE>
__global__ void  _transposeKernel_v1(float* __restrict__ src, float* __restrict__ dst, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float cache[BLOCK_SIZE][BLOCK_SIZE];
    if(row < M && col < N){
        //防止bank冲突，写入的时候就直接转置好
        cache[threadIdx.x][threadIdx.y] = src[row*N+col];
    }
    __syncthreads();
    //块首地址+块内索引
    int dst_row = blockIdx.x * blockDim.x + threadIdx.y;
    int dst_col = blockIdx.y * blockDim.y + threadIdx.x;
    if(dst_row < N && dst_col < M){
        dst[dst_row*M+dst_col] = cache[threadIdx.y][threadIdx.x];
    }
}
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

void compare(float *src1, float* res, int M, int N){
    for(int i = 0; i<N; i++){
        for(int j = 0; j<M; j++){
            if(src1[j*N+i] != res[i*M+j]){
                printf("error at (%d, %d), %f, %f\n", i, j, src1[j*N+i], res[i*M+j]);
                return;
            }
        }
    }
    printf("success!\n");
}
int main(){
    int M = 1024;
    int N = 1024;
    float* host_src = (float*)malloc(M*N*sizeof(float));
    float* host_dst = (float*)malloc(M*N*sizeof(float));
    for(int i = 0; i < M*N; i++){
        host_src[i] = i%32;
    }
    float* device_src, *device_dst;
    cudaMalloc((void**)&device_src, M*N*sizeof(float));
    cudaMalloc((void**)&device_dst, M*N*sizeof(float));
    cudaMemcpy(device_src, host_src, M*N*sizeof(float), cudaMemcpyHostToDevice);
    const int block_size = 32;
    int grid_size_x = (N + block_size - 1) / block_size;
    int grid_size_y = (M + block_size - 1) / block_size;
    _transposeKernel_v1<32><<<dim3(grid_size_x, grid_size_y), dim3(block_size, block_size)>>>(device_src, device_dst, M, N);
    /*================计时================*/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);//将计时开始事件提交到队列
    cudaEventQuery(start);//刷新队列

    for(int i = 0; i<1000; i++){
        _transposeKernel_native<<<dim3(grid_size_x, grid_size_y), dim3(block_size, block_size)>>>(device_src, device_dst, M, N);
    }
    cudaEventRecord(stop);//将计时结束事件提交到队列
    cudaEventSynchronize(stop);//让主机等待stop结束,类似刷新队列
    float elapsed_time;//初始化运行时间
    cudaEventElapsedTime(&elapsed_time, start, stop);//计算运行时间
    printf("Time = %g ms. \n", elapsed_time);

    /*消除计时事件*/
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /*================结束计时================*/

    cudaMemcpy(host_dst, device_dst, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_src);
    cudaFree(device_dst);
    compare(host_src, host_dst, M, N);
    free(host_src);
    free(host_dst);
    return 0;
}