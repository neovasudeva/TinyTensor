#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH  16
#define GEMM_TILE_WIDTH 64


__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    const int m = blockIdx.z;
    const int h = blockIdx.y*TILE_WIDTH + threadIdx.y;
    const int w = blockIdx.x*TILE_WIDTH + threadIdx.x;

    if (h >= H_out || w >= W_out) return;

    for (int b = 0; b < B; b++) {
        float Pvalue = 0.0;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    Pvalue += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = Pvalue;
    }

#undef y4d
#undef x4d
#undef k4d
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int input_size = B*C*H*W;
    const int kernel_size = M*C*K*K;
    const int output_size = B*M*H_out*W_out;
    
    // Declare relevant device pointers
    float* device_x;
    float* device_k;
    float* device_y;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**)&device_x, input_size*sizeof(float));
    cudaMalloc((void**)&device_k, kernel_size*sizeof(float));
    cudaMalloc((void**)&device_y, output_size*sizeof(float));

    cudaMemcpy(device_x, host_x, input_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, kernel_size*sizeof(float), cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    dim3 DimGrid(ceil(1.0*W_out/TILE_WIDTH), ceil(1.0*H_out/TILE_WIDTH), M);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    // Copy the output back to host
    cudaMemcpy(host_y, device_y, output_size*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_k);
    cudaFree(device_y);

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

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

        __syncthreads();
    }

    // load values to output
    if (row < A_rows && col < B_cols)
        output[row * B_cols + col] = pValue;
}

/* entry point for testing */
int main(int argc, char** argv) {
    /*
    std::cout << "GPU DEVICE PROPERTIES" << std::endl;
    GPUInterface i;
    i.get_device_properties();
    */

    // GEMM KERNEL TEST
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

    /*
    for (int i = 0; i < N * M; i++) {
        A[i] = rand() % 10;
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < M * P; i++) {
        B[i] = rand() % 10;
        std::cout << B[i] << " ";
    }
    std::cout << std::endl;
    */

    cudaMalloc((void**) &deviceA, sizeof(float) * N * M);
    cudaMalloc((void**) &deviceB, sizeof(float) * M * P);
    cudaMalloc((void**) &deviceOutput, sizeof(float) * N * P);
    cudaMemcpy(deviceA, A, sizeof(float) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, sizeof(float) * M * P, cudaMemcpyHostToDevice);

    dim3 gridDim(ceil(1.0 * P / GEMM_TILE_WIDTH), ceil(1.0 * N / GEMM_TILE_WIDTH), 1);
    dim3 blockDim(GEMM_TILE_WIDTH, GEMM_TILE_WIDTH, 1);
    gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceOutput, N, M, M, P);
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * N * P, cudaMemcpyDeviceToHost);

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