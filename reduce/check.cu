/***
check.cu用来调试cuda代码保证正确性，不参与python包的封装
***/ 
#include<stdio.h>
template<typename T, int BLOCK_SIZE>
__global__ void sumKernel(const T* src, T* dst, int N, int method=0);

void compare(float *src, float* res, int N){
    float s = 0;
    for(int i = 0; i < N; i++){
        s+=src[i];
    }
    if(s!=*res)
        printf("error is %f\n", *res-s);
    else
        printf("All Close! Resukt is %f\n", s);
}

int main(){
    int N = 2048;
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
    dim3 blocks(block_size);
    int grid_size = (N+block_size-1)/block_size;
    dim3 grid(grid_size);
    sumKernel<float, block_size><<<1, blocks>>>(dsrc, ddst, N, 0);
    cudaMemcpy(hdst, ddst, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dsrc);
    cudaFree(ddst);
    compare(hsrc,hdst,N);
    free(hsrc);
    free(hdst);
    return 0;
}