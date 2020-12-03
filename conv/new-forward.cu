#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH  16
#define GEMM_TILE_WIDTH 32
#define CUDA_MAX_NUM_THREADS 1024
    
__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        
        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

/* helper function to find the nearest power of two greater than num */
int p2up(int num) {
    return pow(2, (int) (log(num) / log(2)) + 1);
}

/* general matrix multiply kernel */
__global__ void gemm(float* A, float* B, float* output, int A_rows, int A_cols, int B_rows, int B_cols) {
    // shared memory for both A and B matrices
    __shared__ float Ads[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
    __shared__ float Bds[GEMM_TILE_WIDTH][GEMM_TILE_WIDTH];
    
    // variables for easy reference
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // each thread loads element from A and B into shared memory
    float pValue = 0;
    for (int ph = 0; ph < (A_cols - 1) / GEMM_TILE_WIDTH + 1; ph++) {
        // load ADS
        if (row < A_rows && (ph * GEMM_TILE_WIDTH + tx) < A_cols) 
            Ads[ty][tx] = A[row * A_cols + (ph * GEMM_TILE_WIDTH) + tx];
        else 
            Ads[ty][tx] = 0;

        // load BDS
        if ((ph * GEMM_TILE_WIDTH + ty) < B_rows && col < B_cols)
            Bds[ty][tx] = B[(ph * GEMM_TILE_WIDTH + ty) * B_cols + col];
        else
            Bds[ty][tx] = 0;

        // wait for data loading to complete and do calculation
        __syncthreads();
        for (int k = 0; k < GEMM_TILE_WIDTH; k++) 
            pValue += Ads[ty][k] * Bds[k][tx];
    }

    // load values to output
    if (row < A_rows && col < B_cols)
        output[row * B_cols + col] = pValue;
}

/* gemm test function */
void test_gemm(char** argv) {
    int N;
    int M;
    int P;
    sscanf(argv[1], "%d", &N); 
    sscanf(argv[2], "%d", &M); 
    sscanf(argv[3], "%d", &P); 

    float* A = new float[N * M];
    float* B = new float[M * P];
    float* hostOutput = new float[N * P];
    float* deviceA;
    float* deviceB;
    float* deviceOutput;

    for (int i = 0; i < N * M; i++) {
        A[i] = rand() % 10;
        //std::cout << A[i] << " ";
    }
    //std::cout << std::endl;
    for (int i = 0; i < M * P; i++) {
        B[i] = rand() % 10;
        //std::cout << B[i] << " ";
    }
    //std::cout << std::endl;

    //cudaMemcpyToSymbol(weight_matrix, B, M * P *sizeof(float));

    cudaMalloc((void**) &deviceA, sizeof(float) * N * M);
    cudaMalloc((void**) &deviceB, sizeof(float) * M * P);
    cudaMalloc((void**) &deviceOutput, sizeof(float) * N * P);
    cudaMemcpy(deviceA, A, sizeof(float) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, sizeof(float) * M * P, cudaMemcpyHostToDevice);

    dim3 gridDim(ceil(1.0 * P / GEMM_TILE_WIDTH), ceil(1.0 * N / GEMM_TILE_WIDTH), 1);
    dim3 blockDim(GEMM_TILE_WIDTH, GEMM_TILE_WIDTH, 1);
    gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceOutput, N, M, M, P);
    cudaDeviceSynchronize(); 
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * N * P, cudaMemcpyDeviceToHost);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    /*
    for (int i = 0; i < N * P; i++) 
        std::cout << hostOutput[i] << " ";
    std::cout << std::endl;
    */

    // free memory
    delete [] A;
    delete [] B;
    delete [] hostOutput;
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceOutput);
}

/* kernel for unrolling 
 * TODO: optimize (maybe shared memory reading from input)
*/
__global__ void unroll(float* input, float* output, int C, int K, int H, int W) {
    // variables used later
    int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    int c, s, h_out, w_out, w_base;

    // verify thread is in bounds of K * K sections        
    if (t < C * W_unroll) {
        // get K * K square to operate on out of all C input feature maps
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;

        // unroll K * K square
        w_base = c * K * K * H_out * W_out;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                output[w_base + (p * K + q) * H_out * W_out + s] = input[c * (H * W) + (h_out + p) * W + w_out + q];
            }
        }
    }
}

/* test function for unrolling ONE image */
void test_unroll() {
    // init input image feature maps
    const int C = 2;
    const int K = 2;
    const int H = 4;
    const int W = 3;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    float hostInput[C * H * W] = {1, 3, 6, 2, 0, 3, 1, 0, 0, 3, 2, 7, 4, 2, 9, 3, 6, 2, 8, 2, 1, 1, 0, 3};
    float* hostOutput = new float[C * K * K * H_out * W_out];
    float* deviceInput;
    float* deviceOutput;

    // initialize device memory
    cudaMalloc((void**) &deviceInput, sizeof(float) * C * H * W);
    cudaMalloc((void**) &deviceOutput, sizeof(float) * C * (K * K) * W_out * H_out);
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * C * H * W, cudaMemcpyHostToDevice);

    // initialize kernel
    dim3 gridDim(ceil((C * H_out * W_out * 1.0) / CUDA_MAX_NUM_THREADS), 1, 1);
    dim3 blockDim(CUDA_MAX_NUM_THREADS, 1, 1);
    unroll<<<gridDim, blockDim>>>(deviceInput, deviceOutput, C, K, H, W);
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * C * H_out * W_out * K * K, cudaMemcpyDeviceToHost);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // check output
    /*
    for (int i = 0; i < C * K * K * H_out * W_out; i++) 
        std::cout << hostOutput[i] << " ";
    std::cout << std::endl;
    */

    // free memory
    delete [] hostOutput;
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

/* 
 * improved unrolling with shared memory and fused with GEMM
 * each block finds entire column in unrolled matrix and each thread loads one element from KxK block C times
 * then does GEMM via a reduction tree
 * weight matrix is in constant memory
 * 
 * blockIdx.x = which KxK block in a given input feature map is taken by these threads, gets whole column in unrolled matrix
 * threadIdx.x = takes an element in the KxK block and stores in shared memory
 * uses dynamic shared memory 
*/
// constant memory for weights in matrix multiplication
__constant__ float weight[5000];

__global__ void unroll_gemm(float* input, float* output, int M, int C, int K, int H, int W) {
    // constants 
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int t = threadIdx.x;
    int b = blockIdx.x;
    int batch = blockIdx.y;
    int c, s, h_out, w_out, h_base, w_base;
    
    // shared memory = one column in unrolled matrix
    // SIZE PER THREAD BLOCK IN BYTES MUST BE SPECIFIED IN KERNEL LAUNCH PARAMETERS
    // size should be C * K * K
    extern __shared__ float dyn_shared[];
    float* unrolled = (float*) dyn_shared;
    float* gemm = (float*) &unrolled[K * K * C]; // nearest power of 2 larger than C * K * K

    // UNROLLING
    if (t < K * K * C) {
        // find feature map to get the K*K block from 
        c = t / (K * K);    
        s = t % (K * K);
        h_out = s / K;
        w_out = s % K;
        h_base = b / W_out;
        w_base = b % W_out;

        // copy data to shared memory
        unrolled[t] = input[batch * (C * H * W) + c * (H * W) + (h_base * W + w_base) + h_out * W + w_out];
    }

    // GEMM via list reduction
    // each block will go find M elements in output (1 col)
    int p2down = (int) powf(2, (int) (logf(C * K * K) / logf(2)));
    int p2up = (int) powf(2, (int) (logf(C * K * K) / logf(2)) + 1);
    for (int m = 0; m < M; m++) {
        // store multplication results in gemm shared memory
        gemm[t] = unrolled[t] * weight[m * (K * K * C) + t];
        if (blockDim.x + t < p2up)
            gemm[blockDim.x + t] = 0;

        // perform list reduction gemm (half of threads)
        for (int stride = p2down; stride >= 1; stride /= 2) { 
            __syncthreads(); 
            if (t < stride) 
                gemm[t] += gemm[t + stride];
        }

        // write output
        if (t == 0) 
            output[batch * (M * H_out * W_out) + m * (H_out * W_out) + b] = gemm[0];
    }
}

void test_unroll_gemm() {
    const int B = 2;
    const int C = 3;
    const int M = 2;
    const int K = 2;
    const int H = 3;
    const int W = 3;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    float hostInput[B * C * H * W] = {1, 2, 0, 1, 1, 3, 0, 2, 2, 
                                  0, 2, 1, 0, 3, 2, 1, 1, 0, 
                                  1, 2, 1, 0, 1, 3, 3, 3, 2,

                                  2, 1, 3, 0, 3, 1, 2, 0, 3,
                                  1, 2, 2, 3, 2, 1, 1, 0, 0, 
                                  1, 3, 2, 0, 0, 2, 3, 1, 1};
    float hostWeight[K * K * M * C] = {1, 1, 2, 2,
                                       1, 1, 1, 1,
                                       0, 1, 1, 0,
                                       1, 0, 0, 1, 
                                       2, 1, 2, 1,
                                       1, 2, 2, 0};
    float* hostOutput = new float[B * M * H_out * W_out];
    //float* hostTemp = new float[C * K * K * H_out * W_out];
    float* deviceInput;
    float* deviceOutput;
    //float* deviceTemp;

    // initialize device memory
    cudaMalloc((void**) &deviceInput, sizeof(float) * B * C * H * W);
    cudaMalloc((void**) &deviceOutput, sizeof(float) * B * M * W_out * H_out);
    //cudaMalloc((void**) &deviceTemp, sizeof(float) * C * K * K * H_out * W_out);
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * B * C * H * W, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(weight, hostWeight, sizeof(float) * K * K * M * C);

    // initialize kernel
    dim3 gridDim(H_out * W_out, B, 1);
    dim3 blockDim(C * K * K, 1, 1);
    int sharedMem = sizeof(float) * C * K * K + sizeof(float) * p2up(C * K * K);
    unroll_gemm<<<gridDim, blockDim, sharedMem>>>(deviceInput, deviceOutput, M, C, K, H, W);
    cudaDeviceSynchronize();
    //cudaMemcpy(hostTemp, deviceTemp, sizeof(float) * C * K * K * H_out * W_out, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * B * M * H_out * W_out, cudaMemcpyDeviceToHost);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // check unrolling
    /*
    for (int i = 0; i < C * K * K; i++) {
        for (int j = 0; j < H_out * W_out; j++) {
            std::cout << hostTemp[i * H_out * W_out + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */

    // check output
    for (int i = 0; i < B * M; i++) {
        for (int j = 0; j < H_out * W_out; j++)
            std::cout << hostOutput[i * H_out * W_out + j] << " ";
        std::cout << std::endl;
    }

    // free memory
    delete [] hostOutput;
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

/* entry point for testing */
int main(int argc, char** argv) {
    // GPU device properties
    GPUInterface i;
    i.get_device_properties();

    /* Kernel sizes to consider:
        B: 100, M: 4, C: 1, K: 7, H: 86, W: 86
        B: 100, M: 16, C: 4, K: 7, H: 40, W: 40
    */

    //test_gemm(argv);
    //test_unroll();
    test_unroll_gemm();
}