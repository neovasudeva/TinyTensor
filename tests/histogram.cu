#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define BLOCK_SIZE 16

/* kernel to convert to unsigned char */
/*
__global__ void ucharConvert(float* input, unsigned char* output, int height, int width, int channels) {
    int row = threadIdx.y + BLOCK_SIZE * blockIdx.y;
    int col = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    if (row < height && col < width) {
        int rgbOffset = (row * width + col) * channels;
        output[rgbOffset] = (unsigned char) (255 * input[rgbOffset]);
        output[rgbOffset + 1] = (unsigned char) (255 * input[rgbOffset + 1]);
        output[rgbOffset + 2] = (unsigned char) (255 * input[rgbOffset + 2]);
    }

}
*/

/* kernel to convert image to unsigned char format */
__global__ void greyscale(float* input, unsigned char* output, int height, int width, int channels) {
    // shared memory
    __shared__ float rgbBlock[BLOCK_SIZE * BLOCK_SIZE * 3];

    // get coordinates
    int col = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    int row = threadIdx.y + BLOCK_SIZE * blockIdx.y;

    // load to shared memory and do computation
    if (row < height && col < width) {
        // offsets
        int greyOffset = row * width + col;

        // load to shared memory
        for (int i = 0; i < channels; i++)
            rgbBlock[(threadIdx.x + BLOCK_SIZE * threadIdx.y) * channels + i] = input[(row * width + col) * channels + i];

        // convert input float value to unsigned char
        unsigned char r = (unsigned char) (255 * rgbBlock[(threadIdx.x + BLOCK_SIZE * threadIdx.y) * channels]);  
        unsigned char g = (unsigned char) (255 * rgbBlock[(threadIdx.x + BLOCK_SIZE * threadIdx.y) * channels + 1]);
        unsigned char b = (unsigned char) (255 * rgbBlock[(threadIdx.x + BLOCK_SIZE * threadIdx.y) * channels + 2]);

        // greyscale
        output[greyOffset] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
    }
}

int main() {
    // init image
    const int height = 3;
    const int width = 3;
    const int channels = 3;
    float imageHost[height * width * channels];

    // fill image array
    for (int i = 0; i < height * width * channels; i++)
        imageHost[i] = (float) rand() / RAND_MAX;

    // intermediate arrays
    float* imageDevice;
    unsigned char* ucharImageDevice;
    unsigned char* greyscaleDevice;
    unsigned char* outputHost;

    // space allocation for intermediate arrays
    cudaMalloc((void**) &imageDevice, sizeof(float) * height * width * channels); 
    cudaMalloc((void**) &ucharImageDevice, sizeof(unsigned char) * height * width * channels);
    cudaMalloc((void**) &greyscaleDevice, sizeof(unsigned char) * height * width);
    cudaMemcpy(imageDevice, imageHost, sizeof(float) * height * width * channels, cudaMemcpyHostToDevice);

    outputHost = (unsigned char*) malloc(sizeof(unsigned char) * height * width);

    // dim sizes
    dim3 blockDim;
    dim3 gridDim;

    // convert to unsigned characters
    /*
    blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    gridDim = dim3(ceil(((float)width) / BLOCK_SIZE), ceil(((float)height / BLOCK_SIZE)), 1);
    ucharConvert<<<gridDim, blockDim>>>(imageDevice, ucharImageDevice, height, width, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(outputHost, ucharImageDevice, sizeof(unsigned char) * height * width * channels, cudaMemcpyDeviceToHost);
    */

    // greyscale the image
    blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    gridDim = dim3(ceil(((float)width) / BLOCK_SIZE), ceil(((float)height / BLOCK_SIZE)), 1);
    greyscale<<<gridDim, blockDim>>>(imageDevice, greyscaleDevice, height, width, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(outputHost, greyscaleDevice, sizeof(unsigned char) * height * width, cudaMemcpyDeviceToHost);

    // create histogram

    // DO NOT COPY OVER: for testing purposes only
    for (int i = 0; i < height * width * channels; i++) 
        std::cout << imageHost[i] * 255 << " ";
    std::cout << std::endl;

    for (int i = 0; i < height * width; i++) 
        std::cout << (int) outputHost[i] << " ";
    std::cout << std::endl;
}