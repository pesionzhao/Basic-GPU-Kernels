#include<stdio.h>
// 共享内存版
template<typename T, int BLOCK_SIZE>
__global__ void GEMMv2(const T* src1, const T* src2, T* dst, int M, int N, int K){
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;
    __shared__ T sa[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ T sb[BLOCK_SIZE*BLOCK_SIZE];
    int loop = (K+BLOCK_SIZE-1)/BLOCK_SIZE;
    T res = 0;
    for(int i = 0; i<loop; i++){
        //存数据
        sa[threadIdx.y*BLOCK_SIZE+threadIdx.x] = src1[row*K+i*BLOCK_SIZE+threadIdx.x];
        sb[threadIdx.y*BLOCK_SIZE+threadIdx.x] = src2[(i*BLOCK_SIZE+threadIdx.y)*N+col];
        __syncthreads();
        //计算
        for(int j = 0; j<BLOCK_SIZE; j++){
            res += sa[threadIdx.y*BLOCK_SIZE+j]*sb[threadIdx.x+j*BLOCK_SIZE];
        }
        __syncthreads();
    }
    dst[row*N+col] = res;
}

// 一个线程操作四个元素
template<typename T, int BLOCK_SIZE, int num_per_thread=4>
__global__ void GEMMv3(const T* src1, const T* src2, T* dst, int M, int N, int K){
    static_assert(num_per_thread>=4&&num_per_thread%4==0, "num_per_thread%4 must equal to 0");
    int col = num_per_thread*blockDim.x*blockIdx.x;
    int row = num_per_thread*blockDim.y*blockIdx.y;
    __shared__ T sa[BLOCK_SIZE*BLOCK_SIZE*num_per_thread];
    __shared__ T sb[BLOCK_SIZE*BLOCK_SIZE*num_per_thread];
    //block中的线程索引
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
    int col_sa = tid%(BLOCK_SIZE/num_per_thread);
    int row_sa = tid/(BLOCK_SIZE/num_per_thread);
    int loop = (K+BLOCK_SIZE-1)/(BLOCK_SIZE);
    T res[num_per_thread*num_per_thread] = {0};
    for(int i = 0; i<loop; i++){
        for(int j = 0; j<num_per_thread; ++j){
            sa[tid*4+j]=src1[(row+row_sa)*K+i*BLOCK_SIZE+col_sa*4+j];
            sb[tid*4+j]=src2[(i*BLOCK_SIZE+threadIdx.y)*N+col+threadIdx.x*4+j];
        }
        __syncthreads();
        //计算
        for(int num_ele = 0; num_ele<num_per_thread*num_per_thread; num_ele++){
            for(int j = 0; j<BLOCK_SIZE; j++){
                res[num_ele] += sa[(threadIdx.y*num_per_thread+num_ele/num_per_thread)*BLOCK_SIZE+j]*sb[threadIdx.x*num_per_thread+num_ele%num_per_thread+j*4*BLOCK_SIZE];
            }
        }
        __syncthreads();
    }
    for(int num_ele = 0; num_ele<num_per_thread*num_per_thread; num_ele++){
        dst[(row+threadIdx.y*num_per_thread+num_ele/num_per_thread)*N+num_ele%num_per_thread+col+threadIdx.x*num_per_thread] = res[num_ele];
    }
}
//float4
// 一个线程操作四个元素
template<typename T, int BLOCK_SIZE, int num_per_thread=4>
__global__ void GEMMv4(const T* src1, const T* src2, T* dst, int M, int N, int K){
    static_assert(num_per_thread>=4&&num_per_thread%4==0, "num_per_thread%4 must equal to 0");
    int col = num_per_thread*blockDim.x*blockIdx.x;
    int row = num_per_thread*blockDim.y*blockIdx.y;
    __shared__ T sa[BLOCK_SIZE*BLOCK_SIZE*num_per_thread];
    __shared__ T sb[BLOCK_SIZE*BLOCK_SIZE*num_per_thread];
    //block中的线程索引
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
    int col_sa = tid%(BLOCK_SIZE/num_per_thread);
    int row_sa = tid/(BLOCK_SIZE/num_per_thread);
    int loop = (K+BLOCK_SIZE-1)/(BLOCK_SIZE);
    T res[num_per_thread*num_per_thread] = {0};
    for(int i = 0; i<loop; i++){
        (float4&)sa[tid*4]=(float4&)src1[(row+row_sa)*K+i*BLOCK_SIZE+col_sa*4];
        (float4&)sb[tid*4]=(float4&)src2[(i*BLOCK_SIZE+threadIdx.y)*N+col+threadIdx.x*4];
        __syncthreads();
        //计算
        for(int num_ele = 0; num_ele<num_per_thread*num_per_thread; num_ele++){
            for(int j = 0; j<BLOCK_SIZE; j++){
                res[num_ele] += sa[(threadIdx.y*num_per_thread+num_ele/num_per_thread)*BLOCK_SIZE+j]*sb[threadIdx.x*num_per_thread+num_ele%num_per_thread+j*4*BLOCK_SIZE];
            }
        }
        __syncthreads();
    }
    for(int num_ele = 0; num_ele<num_per_thread*num_per_thread; num_ele++){
        dst[(row+threadIdx.y*num_per_thread+num_ele/num_per_thread)*N+num_ele%num_per_thread+col+threadIdx.x*num_per_thread] = res[num_ele];
    }
}
//转置防止写入的bank冲突
template<typename T, int BLOCK_SIZE, int num_per_thread=4>
__global__ void GEMMv5(const T* src1, const T* src2, T* dst, int M, int N, int K){
    static_assert(num_per_thread>=4&&num_per_thread%4==0, "num_per_thread%4 must equal to 0");
    int col = num_per_thread*blockDim.x*blockIdx.x;
    int row = num_per_thread*blockDim.y*blockIdx.y;
    __shared__ T sa[BLOCK_SIZE*BLOCK_SIZE*num_per_thread];
    __shared__ T sb[BLOCK_SIZE*BLOCK_SIZE*num_per_thread];
    //block中的线程索引
    int tid = threadIdx.y*blockDim.x+threadIdx.x;
    int col_sa = tid%(BLOCK_SIZE/num_per_thread);
    int row_sa = tid/(BLOCK_SIZE/num_per_thread);
    int col_sb = tid%BLOCK_SIZE;
    int row_sb = tid/BLOCK_SIZE;
    int loop = (K+BLOCK_SIZE-1)/(BLOCK_SIZE);
    T res[num_per_thread*num_per_thread] = {0};
    for(int i = 0; i<loop; i++){
        //写入时防止bank冲突，导致了都被转置保存
        float4 tmp = (float4&)src1[(row+row_sa)*K+i*BLOCK_SIZE+col_sa*4];
        sa[col_sa*4*BLOCK_SIZE*num_per_thread+row_sa] = tmp.x; 
        sa[(col_sa*4+1)*BLOCK_SIZE*num_per_thread+row_sa] = tmp.y; 
        sa[(col_sa*4+2)*BLOCK_SIZE*num_per_thread+row_sa] = tmp.z; 
        sa[(col_sa*4+3)*BLOCK_SIZE*num_per_thread+row_sa] = tmp.w; 
        tmp = (float4&)src2[(i*BLOCK_SIZE+threadIdx.y)*N+col+threadIdx.x*4];
        sb[col_sb*4*BLOCK_SIZE+row_sb] = tmp.x; 
        sb[(col_sb*4+1)*BLOCK_SIZE+row_sb] = tmp.y; 
        sb[(col_sb*4+2)*BLOCK_SIZE+row_sb] = tmp.z; 
        sb[(col_sb*4+3)*BLOCK_SIZE+row_sb] = tmp.w; 
        __syncthreads();
        //计算
        for(int num_ele = 0; num_ele<num_per_thread*num_per_thread; num_ele++){
            for(int j = 0; j<BLOCK_SIZE; j++){
                res[num_ele] += sa[(threadIdx.y*num_per_thread+num_ele/num_per_thread)+j*BLOCK_SIZE*4]*sb[(threadIdx.x*num_per_thread+num_ele%num_per_thread)*BLOCK_SIZE+j];
            }
        }
        __syncthreads();
    }
    for(int num_ele = 0; num_ele<num_per_thread*num_per_thread; num_ele++){
        dst[(row+threadIdx.y*num_per_thread+num_ele/num_per_thread)*N+num_ele%num_per_thread+col+threadIdx.x*num_per_thread] = res[num_ele];
    }
}

template <typename T>
void launchGEMMv2Kernel(const T* src1,const T* src2,T* dst,int M,int N,int K){
    const int block_size_x = 32;
    const int block_size_y = 32;
    int grid_size_x = (N+block_size_x-1)/(block_size_x);
    int grid_size_y = (M+block_size_y-1)/(block_size_y);
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);
    GEMMv2<float, block_size_x><<<grid,block>>>(src1, src2, dst, M, N, K);
}
template <typename KernelFunc, typename T, int BLOCK_SIZE>
void launchGEMMKernel(const T* src1,const T* src2,T* dst,int M,int N,int K, KernelFunc kernel){
    const int block_size_x = BLOCK_SIZE;
    const int block_size_y = BLOCK_SIZE;
    int num_per_threads = 4;
    int grid_size_x = (N+block_size_x*num_per_threads-1)/(block_size_x*num_per_threads);
    int grid_size_y = (M+block_size_y*num_per_threads-1)/(block_size_y*num_per_threads);
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);
    kernel<<<grid,block>>>(src1, src2, dst, M, N, K); //调用时没有模板, 指定KernelFunc时需要指定模板
}

// 必须指定kernel的模板
template <typename T>
void launchGEMMv3(const T* src1,const T* src2,T* dst,int M,int N,int K){
    const int BLOCK_SIZE = 32;
    launchGEMMKernel<decltype(GEMMv3<T, BLOCK_SIZE>), T, BLOCK_SIZE>(src1, src2, dst, M, N, K, GEMMv3<T, BLOCK_SIZE>);
}
template <typename T>
void launchGEMMv4(const T* src1,const T* src2,T* dst,int M,int N,int K){
    const int BLOCK_SIZE = 32;
    launchGEMMKernel<decltype(GEMMv4<T, BLOCK_SIZE>), T, BLOCK_SIZE>(src1, src2, dst, M, N, K, GEMMv4<T, BLOCK_SIZE>);
}
template <typename T>
void launchGEMMv5(const T* src1,const T* src2,T* dst,int M,int N,int K){
    const int BLOCK_SIZE = 32;
    launchGEMMKernel<decltype(GEMMv5<T, BLOCK_SIZE>), T, BLOCK_SIZE>(src1, src2, dst, M, N, K, GEMMv5<T, BLOCK_SIZE>);
}