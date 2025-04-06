#include<stdio.h>
#include"softmax.cuh"

void compare(float *src, float* res, int M, int N){
    float* dst = new float[N*M]; 
    for(int row = 0; row<M; row++){
        float global_sum = 0;
        float global_max = -1000;
        for(int i = 0; i < N; i++){
            global_max = max(global_max, src[i+row*N]);
        }
        for(int i = 0; i < N; i++){
            dst[i+row*N] = exp(src[i+row*N]-global_max);
            global_sum+=dst[i+row*N];
        }
        for(int i = 0; i<N; i++){
            dst[i+row*N] /= global_sum;
            if(abs(dst[i+row*N]-res[i+row*N])>1e-6){
                printf("error occurs between %f and %f in [%d, %d] src = %f\n", dst[i+row*N], res[i+row*N], row, i, src[i+row*N]);
                return;
            }
        }
    }
    printf("All Close!\n");
}
// 只有N<=1024的情况使用
void launchSoftMax2dwarp(const float* src, float* dst, int M, int N){
    const int block_size_x = 32;
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

int main(){
    int N = 4096;
    int M = 8;
    float *hsrc = (float *)malloc(N * M * sizeof(float));
    float *hdst = (float *)malloc(N * M * sizeof(float));
    for(int i = 0; i < N*M; i++){
        hsrc[i] = (i%37)*0.1;
    }
    float *dsrc;
    cudaMalloc((void **)&dsrc, N * M * sizeof(float));
    cudaMemcpy(dsrc, hsrc, N * M * sizeof(float), cudaMemcpyHostToDevice);
    float *ddst;
    cudaMalloc((void **)&ddst, N * M *sizeof(float));
    if(N<=1024)
        launchSoftMax2dwarp(dsrc, ddst, M, N);
    else
        launchSoftMax2dShared(dsrc, ddst, M, N);
    cudaMemcpy(hdst, ddst, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dsrc);
    cudaFree(ddst);
    compare(hsrc,hdst,M,N);
    free(hsrc);
    free(hdst);
    return 0;
}