#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256

using namespace std;

__global__ void vectorInc(int* input, int* output, int numElements) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    // bounds
    if (t < numElements) 
        output[t] = input[t] + 1;
}

// entry point
int main() {
    const int numElements = 10;
    int host_input[numElements] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int* device_input;
    int* device_output;
    int* host_output;

    // copy memory and allocate for output
    host_output = (int*) malloc(sizeof(int) * numElements);
    cudaMalloc((void**) &device_input, sizeof(int) * numElements);
    cudaMalloc((void**) &device_output, sizeof(int) * numElements);
    cudaMemcpy(device_input, host_input, sizeof(int) * numElements, cudaMemcpyHostToDevice);

    // init kernel
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(1.0 * numElements / BLOCK_SIZE));
    vectorInc<<<gridDim, blockDim>>>(device_input, device_output, numElements);

    // wait for device to finish
    cudaDeviceSynchronize();

    // copy answer back to host
    cudaMemcpy(host_output, device_output, sizeof(int) * numElements, cudaMemcpyDeviceToHost);

    // verify answer
    for (int i = 0; i < numElements; i++) {
        cout << host_output[i] << " ";
    }
    cout << endl;

    // free memory 
    cudaFree(device_output);
    cudaFree(device_input);
    free(device_input);
    free(device_output);

    return 0;
}