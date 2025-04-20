#include<stdio.h>
#include <random>
#include"FA.cuh"
#include"gemm.cuh"
#include"rand.h"
void compare(float* a, float* b, int M, int N){
    int cnt = 0;
    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            if(abs(a[i*N+j]-b[i*N+j])>1e-6){
                printf("error occurs in[%d][%d], cpu_ans = %f, fa = %f\n", i, j, a[i*N+j], b[i*N+j]);
                cnt++;
                if (cnt==32)
                    return;
            }
        }
    }
    printf("All close!!\n");
    printf("result in[0][0] is %f\n", a[200]);
    return;
}
void attention(float* Q, float* K, float* V, float* dst,  int seq_len, int head_dim){
    //q*k^T
    float* O = new float[seq_len*seq_len]();
    for(int i = 0; i<seq_len; i++){
        for(int j = 0; j<seq_len; j++){
            for(int k = 0; k<head_dim; k++){
                // O[i*seq_len+j] += Q[i*head_dim+k]*K[k*seq_len+j]; //q*k
                O[i*seq_len+j] += Q[i*head_dim+k]*K[k+head_dim*j]; //q*k^T
            }
        }
    }
    //softmax
    for(int i = 0; i<seq_len; i++){
        float max_v = -100;
        float sum_v = 0.0;
        for(int j = 0; j<seq_len; j++){
            max_v = max(max_v, O[i*seq_len+j]);
        }
        for(int j = 0; j<seq_len; j++){
            O[i*seq_len+j] = exp(O[i*seq_len+j]-max_v);
            sum_v += O[i*seq_len+j];
        }
        for(int j = 0; j<seq_len; j++){
            O[i*seq_len+j] /= sum_v;
        }
        // printf("%d row max = %f, sum = %f\n", i, max_v, sum_v);
        // printf("%d row O[row][0] = %f\n", i, O[i*seq_len]);
    }
    //O*V
    for(int i = 0; i<seq_len; i++){
        for(int j = 0; j<head_dim; j++){
            for(int k = 0; k<seq_len; k++){
                dst[i*head_dim+j] += O[i*seq_len+k]*V[k*head_dim+j];
            }
        }
    }
    delete[] O;
}
int main(){
    mt19937_state rng;
    manual_seed(&rng, 1142);
    const int M = 64;
    const int N = 32;
    float* q = (float*)malloc(M*N*sizeof(float));
    float* k = (float*)malloc(M*N*sizeof(float));
    float* v = (float*)malloc(M*N*sizeof(float));
    float* dst = (float*)malloc(M*N*sizeof(float));
    float* fadst = (float*)malloc(M*N*sizeof(float));
    uniform_(q, M*N, 0, 1, &rng);
    uniform_(k, M*N, -0, 1, &rng);
    uniform_(v, M*N, -0, 1, &rng);
    printf("random number is%f\n", q[0]);
    printf("random number is%f\n", k[0]);
    printf("random number is%f\n", v[0]);
    for(int i = 0; i<M; i++){
        for(int j = 0; j<N; j++){
            dst[i*N+j] = 0;
            fadst[i*N+j] = 0;
        }
    }
    float* dq, *dk, *dv, *res, *fares;
    cudaMalloc((void**)&dq, M*N*sizeof(float));
    cudaMalloc((void**)&dk, M*N*sizeof(float));
    cudaMalloc((void**)&dv, M*N*sizeof(float));
    cudaMalloc((void**)&res, M*N*sizeof(float));
    cudaMalloc((void**)&fares, M*N*sizeof(float));
    cudaMemcpy(dq, q, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, M*N*sizeof(float), cudaMemcpyHostToDevice);
    lauchFA<N>(dq, dk, dv, fares, 1,1, M, N);
    cudaMemcpy(fadst, fares, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    // launchGEMMv2Kernel<float>(dq, dk, res, M, N, N);
    // cudaMemcpy(dq, res, M*N*sizeof(float), cudaMemcpyDeviceToDevice);
    // launchGEMMv2Kernel<float>(dq, dv, res, M, N, N);
    // cudaFree(dq);
    // cudaFree(dk);
    // cudaFree(dv);
    // cudaMemcpy(dst, res, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaFree(res);
    // cudaFree(fares);
    attention(q, k, v, dst, M, N);
    compare(dst, fadst, M, N);
    free(q);
    free(k);
    free(v);
    free(dst);
    free(fadst);
}