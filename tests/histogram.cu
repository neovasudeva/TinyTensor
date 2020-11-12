#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define HISTOGRAM_LENGTH 256

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

/* kernel to generate histogram */
__global__ void histogram(unsigned char* image, int* output, int height, int width) {
    // shared memory 
    __shared__ int histo[HISTOGRAM_LENGTH];

    // initialize shared memory to 0
    int t = threadIdx.x + threadIdx.y * BLOCK_SIZE;
    if (t < HISTOGRAM_LENGTH)
        histo[t] = 0;

    // allow threads to finish putting in 0's to histo
    __syncthreads();
    
    // indices in image
    int row = threadIdx.y + BLOCK_SIZE * blockIdx.y;
    int col = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    // increment data to histogram
    if (row < height && col < width) {
        unsigned char pixel = image[row * width + col];
        atomicAdd(&histo[pixel], 1);
    }

    // allow threads in block to finish incrementing data in histo
    __syncthreads();

    // increment output with histo's data for this block
    if (t < HISTOGRAM_LENGTH)
        atomicAdd(&output[t], histo[t]);
}

/* scan kernel to generate CDF of histogram */
__global__ void CDF(int* histogram, float* output, int height, int width) {
    // shared memory 
    __shared__ float cdf[HISTOGRAM_LENGTH];

    // load data to shared memory
    cdf[threadIdx.x] = (float) histogram[threadIdx.x];
    cdf[blockDim.x + threadIdx.x] = (float) histogram[blockDim.x + threadIdx.x];

    // reduction step
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < 2 * blockDim.x) 
            cdf[index] += cdf[index - stride];
    }

    // post-scan step
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x * 2) 
            cdf[index + stride] += cdf[index];
    }

    // write result to output
    __syncthreads();
    output[threadIdx.x] = cdf[threadIdx.x] / (width * height);
    output[threadIdx.x + blockDim.x] = cdf[threadIdx.x + blockDim.x] / (width * height);
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
    int* histogramDevice;
    float* CDFDevice;
    float* outputHost;

    // space allocation for intermediate arrays
    cudaMalloc((void**) &imageDevice, sizeof(float) * height * width * channels); 
    cudaMalloc((void**) &ucharImageDevice, sizeof(unsigned char) * height * width * channels);
    cudaMalloc((void**) &greyscaleDevice, sizeof(unsigned char) * height * width);
    cudaMalloc((void**) &histogramDevice, sizeof(int) * HISTOGRAM_LENGTH);
    cudaMalloc((void**) &CDFDevice, sizeof(float) * HISTOGRAM_LENGTH);

    // assign data 
    cudaMemcpy(imageDevice, imageHost, sizeof(float) * height * width * channels, cudaMemcpyHostToDevice);
    cudaMemset(histogramDevice, 0, sizeof(int) * HISTOGRAM_LENGTH);

    // allocate memory for output
    outputHost = (float*) malloc(sizeof(float) * HISTOGRAM_LENGTH);

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
    //cudaMemcpy(outputHost, greyscaleDevice, sizeof(unsigned char) * height * width, cudaMemcpyDeviceToHost);

    // create histogram
    blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    gridDim = dim3(ceil(((float)width) / BLOCK_SIZE), ceil(((float)height / BLOCK_SIZE)), 1);
    histogram<<<gridDim, blockDim>>>(greyscaleDevice, histogramDevice, height, width);
    cudaDeviceSynchronize();
    //cudaMemcpy(outputHost, histogramDevice, sizeof(int) * HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost);

    // create CDF from histogram
    blockDim = dim3(HISTOGRAM_LENGTH / 2);
    gridDim = dim3(1, 1, 1);
    CDF<<<gridDim, blockDim>>>(histogramDevice, CDFDevice, height, width);
    cudaDeviceSynchronize();
    cudaMemcpy(outputHost, CDFDevice, sizeof(float) * HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost);

    // DO NOT COPY OVER: for testing purposes only
    for (int i = 0; i < height * width * channels; i++) 
        std::cout << imageHost[i] * 255 << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < HISTOGRAM_LENGTH; i++) 
        std::cout << outputHost[i] << " ";
    std::cout << std::endl;
}