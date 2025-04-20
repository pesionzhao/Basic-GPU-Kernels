#include<stdio.h>
#include <random>
#include <unistd.h>  // 引入此头文件来使用 sleep
#include"gemm.cuh"
template<typename T>
void gemmcpu(T* src1, T* src2, T* dst, int M, int N, int K){
    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<K; k++){
                dst[i*N+j] += src1[i*K+k]*src2[k*N+j];
                // if(i*N+j==11)
                // printf("i = %d , j = %d, k = %d, val is %f, %f, dst = %f\n", i,j,k,src1[i*K+k],src2[k*N+j],dst[i*N+j]);
            }
        }
    }
    printf("src in[0][0] is %f\n", src1[200]);
    printf("src in[0][0] is %f\n", src2[200]);
    printf("dst in[0][0] is %f\n", dst[200]);
}
template<typename T>
void compare(T* a, T* b, int M, int N){
    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            if(abs(a[i*N+j]-b[i*N+j])>1e-4){
                printf("error occurs in[%d][%d], cuda = %f, cpu = %f\n", i, j, a[i*N+j], b[i*N+j]);
                return;
            }
        }
    }
    printf("All close!!\n");
    printf("result in[0][0] is %f\n", b[200]);
    return;
}


int main(){
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister 随机数生成器
    float mean = 0.0, stddev = 0.1;  // 均值和标准差
    std::normal_distribution<> dis(mean, stddev);  // 正态分布
    // 生成一个正态分布随机数
    printf("random number is%f\n", dis(gen));
    int M = 256;
    int N = 256;
    int K = 256;
    float* src1 = (float*)malloc(M*K*sizeof(float));
    float* src2 = (float*)malloc(N*K*sizeof(float));
    float* dst = (float*)malloc(M*N*sizeof(float));
    float* dstcpu = (float*)malloc(M*N*sizeof(float)); //需要初始化
    for(int i = 0; i<M; i++){
        for(int j = 0; j<K; j++){
            src1[i*K+j] = dis(gen);
        }
    }
    for(int i = 0; i<K; i++){
        for(int j = 0; j<N; j++){
            src2[i*N+j] = dis(gen);
        }
    }
    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            dst[i*N+j] = 0;
            dstcpu[i*N+j] = 0;
        }
    }
    float* da, *db, *dc;
    cudaMalloc((void**)&da, M*K*sizeof(float));
    cudaMalloc((void**)&db, N*K*sizeof(float));
    cudaMalloc((void**)&dc, M*N*sizeof(float));
    cudaMemcpy(da, src1, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, src2, N*K*sizeof(float), cudaMemcpyHostToDevice);
    launchGEMMv3<float>(da, db, dc, M, N, K);
    cudaFree(da);
    cudaFree(db);
    cudaMemcpy(dst, dc, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dc);
    gemmcpu(src1, src2, dstcpu, M, N, K);
    compare(dst, dstcpu, M, N);
    free(src1);
    free(src2);
    free(dst);
    free(dstcpu);
}