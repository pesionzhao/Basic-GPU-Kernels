/***
check.cu用来调试cuda代码保证正确性，不参与python包的封装
***/ 
#include<stdio.h>
__global__ void _addKernel(float *src1, float *src2, float *dst, int numel);

void compare(float *src1, float *src2, float* res, int M, int N){
    for(int i = 0; i < M * N; i++){
        if(src1[i]+src2[i]-res[i]>1e-5){
            printf("Error occurres! \n");
            return;
        }
    }
    printf("All Close! \n");
}

int main(){
    int M = 1000;
    int N = 1000;
    float *hsrc1 = (float *)malloc(M * N * sizeof(float));
    float *hsrc2 = (float *)malloc(M * N * sizeof(float));
    float *hdst = (float *)malloc(M * N * sizeof(float));
    for(int i = 0; i < M * N; i++){
        hsrc1[i] = i%29;
        hsrc2[i] = i%37;
    }
    float *dsrc1;
    cudaMalloc((void **)&dsrc1, M * N * sizeof(float));
    cudaMemcpy(dsrc1, hsrc1, M * N * sizeof(float), cudaMemcpyHostToDevice);
    float *dsrc2;
    cudaMalloc((void **)&dsrc2, M * N * sizeof(float));
    cudaMemcpy(dsrc2, hsrc2, M * N * sizeof(float), cudaMemcpyHostToDevice);
    float *ddst;
    cudaMalloc((void **)&ddst, M * N * sizeof(float));
    dim3 blocks(32,4);
    dim3 grid(65536,1);
    _addKernel<<<grid, blocks>>>(dsrc1, dsrc2, ddst, M*N);
    cudaMemcpy(hdst, ddst, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dsrc1);
    cudaFree(dsrc2);
    cudaFree(ddst);
    compare(hsrc1,hsrc2,hdst,M,N);
    free(hsrc1);
    free(hsrc2);
    free(hdst);
    return 0;
}