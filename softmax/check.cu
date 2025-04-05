#include<stdio.h>

template<int cols_per_thread, int BLOCK_SIZE>
__global__ void softmax(float* src, float* dst, int N);

template<int cols_per_thread, int BLOCK_SIZE>
__global__ void softmax_native(float* src, float* dst, int N);

void compare(float *src, float* res, int N){
    float* dst = new float[N]; 
    float global_sum = 0;
    float global_max = -1000;
    for(int i = 0; i < N; i++){
        global_max = max(global_max, src[i]);
    }
    for(int i = 0; i < N; i++){
        dst[i] = exp(src[i]-global_max);
        global_sum+=dst[i];
    }
    for(int i = 0; i<N; i++){
        dst[i] /= global_sum;
        if(dst[i]-res[i]>1e-7){
            printf("error occurs between %f and %f\n", dst[i], res[i]);
            return;
        }
    }
    printf("All Close!\n");
}

int main(){
    const int N = 2048;
    float *hsrc = (float *)malloc(N * sizeof(float));
    float *hdst = (float *)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        hsrc[i] = i%48;
    }
    float *dsrc;
    cudaMalloc((void **)&dsrc, N * sizeof(float));
    cudaMemcpy(dsrc, hsrc, N * sizeof(float), cudaMemcpyHostToDevice);
    float *ddst;
    cudaMalloc((void **)&ddst, N * sizeof(float));
    const int block_size = 128;
    constexpr int col_per_thread = (N+block_size-1)/block_size;
    dim3 blocks(block_size);
    int grid_size = (N+block_size-1)/block_size;
    dim3 grid(grid_size);
    softmax<col_per_thread, block_size><<<1, blocks>>>(dsrc, ddst, N);
    cudaMemcpy(hdst, ddst, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dsrc);
    cudaFree(ddst);
    compare(hsrc,hdst,N);
    free(hsrc);
    free(hdst);
    return 0;
}