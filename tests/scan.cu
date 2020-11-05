#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 2 //@@ You can change this

__global__ void scan(float *input, float *output, int len, int layer) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  // normal scan vs interScan?
  // normal scan will perform Brent-Keung on all blocks
  // interScan will then take last element from each block and do Brent-Keung scan on that
  int layerIdx;
  int layerLoad;
  if (!layer) {
    layerIdx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    layerLoad = BLOCK_SIZE;
  } else {
    layerIdx = (threadIdx.x + 1) * 2 * BLOCK_SIZE - 1;
    layerLoad = BLOCK_SIZE * 2;
  }
  // shared memory
  __shared__ float T[2 * BLOCK_SIZE];

  // load data to shared memory
  if (layerIdx < len) 
    T[threadIdx.x] = input[layerIdx];
  else 
    T[threadIdx.x] = 0;
  if (layerIdx + layerLoad < len) 
    T[threadIdx.x + BLOCK_SIZE] = input[layerIdx + layerLoad];
  else 
    T[threadIdx.x + BLOCK_SIZE] = 0;

  // reduction step
  for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < 2 * BLOCK_SIZE /*&& index - stride >= 0*/) {
      T[index] += T[index - stride];
    }
  }

  // post-scan step
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < BLOCK_SIZE * 2) {
      T[index + stride] += T[index];
    }
  }
  
  // write shared memory to output
  __syncthreads();
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) 
    output[i] = T[threadIdx.x];
  if (i+blockDim.x < len) 
    output[i+blockDim.x] = T[threadIdx.x+blockDim.x];
  
}

__global__ void add(float* interScans, float* interSums, float* deviceOutput, int len) {
  // Given the array of scan sums, this kernel will add the interSums to interScans
  
  int t = threadIdx.x + (blockDim.x * blockIdx.x * 2);
  
  // dont do anything for block 0
  if (blockIdx.x != 0) {
    if (t < len) 
      deviceOutput[t] = interScans[t] + interSums[blockIdx.x - 1];
    if (t + BLOCK_SIZE < len)
      deviceOutput[t + BLOCK_SIZE] = interScans[t + BLOCK_SIZE] + interSums[blockIdx.x - 1];
  } else {
    if (t < len) 
      deviceOutput[t] = interScans[t];
    if (t + BLOCK_SIZE < len)
      deviceOutput[t + BLOCK_SIZE] = interScans[t + BLOCK_SIZE];
  }
}

int main(int argc, char **argv) {
    //float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    float *interScans;  // first scan output
    float *interSums;  // sums from first scan
    //int numElements; // number of elements in the list
    float hostInput[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int numElements = 10;

    cudaMalloc((void **)&deviceInput, numElements * sizeof(float));
    cudaMalloc((void **)&deviceOutput, numElements * sizeof(float));
    cudaMalloc((void **)&interScans, numElements * sizeof(float));
    cudaMalloc((void **)&interSums, ceil(numElements / (2.0 * BLOCK_SIZE)) * sizeof(float)); //fix

    cudaMemset(deviceOutput, 0, numElements * sizeof(float));
    cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(numElements / (2.0 * BLOCK_SIZE)));

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<gridDim, blockDim>>>(deviceInput, interScans, numElements, 0);
    cudaDeviceSynchronize();

    float* host_interScans = new float[numElements];
    cudaMemcpy(host_interScans, interScans, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numElements; i++) {
        std::cout << host_interScans[i] << " ";
    }
    std::cout << std::endl;

    scan<<<1, blockDim>>>(interScans, interSums, numElements, 1);
    cudaDeviceSynchronize();

    float* host_interSums = new float[ceil(numElements / (2.0 * BLOCK_SIZE))];
    cudaMemcpy(host_interSums, interSums, ceil(numElements / (2.0 * BLOCK_SIZE)) * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < ceil(numElements / (2.0 * BLOCK_SIZE)); i++) {
        std::cout << host_interSums[i] << " ";
    }
    std::cout << std::endl;

    add<<<gridDim, blockDim>>>(interScans, interSums, deviceOutput, numElements);
    cudaDeviceSynchronize();
    //cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    hostOutput = new float[numElements];
    cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numElements; i++) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(interSums);
    cudaFree(interScans);

    free(hostInput);
    free(hostOutput);

    return 0;
}