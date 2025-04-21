#include<stdio.h>
#include <random>
#include"FA.cuh"
#include"gemm.cuh"
#include"rand.h"
#include <chrono>
void compare(float* a, float* b, int batch, int num_head, int seq_len, int head_dim){
    int cnt = 0;
    for(int bc = 0; bc<batch*num_head; bc++){
        float* a1 = a+bc*seq_len*head_dim;
        float* b1 = b+bc*seq_len*head_dim;
        for(int i = 0; i<seq_len; i++){
            for(int j = 0; j<head_dim; j++){
                if(abs(a1[i*head_dim+j]-b1[i*head_dim+j])>1e-6){
                    printf("%d bc error occurs in[%d][%d], cpu_ans = %f, fa = %f\n", bc, i, j, a1[i*head_dim+j], b1[i*head_dim+j]);
                    cnt++;
                    if (cnt==32)
                        return;
                }
            }
        }
    }
    printf("All close!!\n");
    printf("result in[0][0] is %f\n", a[200]);
    return;
}
void attention(float* Q, float* K, float* V, float* dst,  int batch, int num_head, int seq_len, int head_dim){
    float* O = new float[seq_len*seq_len];
    for(int bc = 0; bc<batch*num_head; bc++){
        memset(O, 0, seq_len*seq_len*sizeof(float));
        float* Q1 = Q+bc*seq_len*head_dim;
        float* K1 = K+bc*seq_len*head_dim;
        float* V1 = V+bc*seq_len*head_dim;
        float* dst1 = dst+bc*seq_len*head_dim;
        //q*k^T
        for(int i = 0; i<seq_len; i++){
            for(int j = 0; j<seq_len; j++){
                for(int k = 0; k<head_dim; k++){
                    // O[i*seq_len+j] += Q[i*head_dim+k]*K[k*seq_len+j]; //q*k
                    O[i*seq_len+j] += Q1[i*head_dim+k]*K1[k+head_dim*j]; //q*k^T
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
                    dst1[i*head_dim+j] += O[i*seq_len+k]*V1[k*head_dim+j];
                }
            }
        }
    }
    delete[] O;
}
int main(){
    mt19937_state rng;
    manual_seed(&rng, 1142);
    int repeat = 100;
    const int batch = 16;
    const int num_head = 16;
    const int seq_len = 64;
    const int head_dim = 32;
    float* q = (float*)malloc(batch*num_head*seq_len*head_dim*sizeof(float));
    float* k = (float*)malloc(batch*num_head*seq_len*head_dim*sizeof(float));
    float* v = (float*)malloc(batch*num_head*seq_len*head_dim*sizeof(float));
    float* dst = (float*)malloc(batch*num_head*seq_len*head_dim*sizeof(float));
    float* fadst = (float*)malloc(batch*num_head*seq_len*head_dim*sizeof(float));
    uniform_(q, batch*num_head*seq_len*head_dim, 0, 1, &rng);
    uniform_(k, batch*num_head*seq_len*head_dim, -0, 1, &rng);
    uniform_(v, batch*num_head*seq_len*head_dim, -0, 1, &rng);
    printf("random number is%f\n", q[0]);
    printf("random number is%f\n", k[0]);
    printf("random number is%f\n", v[0]);
    for(int i = 0; i<seq_len; i++){
        for(int j = 0; j<head_dim; j++){
            dst[i*head_dim+j] = 0;
            fadst[i*head_dim+j] = 0;
        }
    }
    float* dq, *dk, *dv, *res, *fares;
    cudaMalloc((void**)&dq, batch*num_head*seq_len*head_dim*sizeof(float));
    cudaMalloc((void**)&dk, batch*num_head*seq_len*head_dim*sizeof(float));
    cudaMalloc((void**)&dv, batch*num_head*seq_len*head_dim*sizeof(float));
    cudaMalloc((void**)&res, batch*num_head*seq_len*head_dim*sizeof(float));
    cudaMalloc((void**)&fares, batch*num_head*seq_len*head_dim*sizeof(float));
    cudaMemcpy(dq, q, batch*num_head*seq_len*head_dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k, batch*num_head*seq_len*head_dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, batch*num_head*seq_len*head_dim*sizeof(float), cudaMemcpyHostToDevice);
    lauchFA<head_dim>(dq, dk, dv, fares, batch, num_head, seq_len, head_dim);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i<repeat; i++){
        lauchFA<head_dim>(dq, dk, dv, fares, batch, num_head, seq_len, head_dim);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;
    cudaMemcpy(fadst, fares, batch*num_head*seq_len*head_dim*sizeof(float), cudaMemcpyDeviceToHost);
    // launchGEMMv2Kernel<float>(dq, dk, res, seq_len, head_dim, head_dim);
    // cudaMemcpy(dq, res, seq_len*head_dim*sizeof(float), cudaMemcpyDeviceToDevice);
    // launchGEMMv2Kernel<float>(dq, dv, res, seq_len, head_dim, head_dim);
    // cudaFree(dq);
    // cudaFree(dk);
    // cudaFree(dv);
    // cudaMemcpy(dst, res, seq_len*head_dim*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaFree(res);
    // cudaFree(fares);
    attention(q, k, v, dst, batch, num_head, seq_len, head_dim);
    compare(dst, fadst, batch, num_head, seq_len, head_dim);
    free(q);
    free(k);
    free(v);
    free(dst);
    free(fadst);
}