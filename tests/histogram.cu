#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define HISTOGRAM_LENGTH 256

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

/* trash kernel to generate histogram */
__global__ void trash_histogram(unsigned char* image, int* output, int height, int width) {
    // indices in image
    int row = threadIdx.y + BLOCK_SIZE * blockIdx.y;
    int col = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    // increment data to histogram
    if (row < height && col < width) {
        unsigned char pixel = image[row * width + col];
        atomicAdd(&output[pixel], 1);
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

/* kernel to do equalization */
__global__ void equalization(float* input, float* cdf, float* output, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    // do equalization
    if (row < height && col < width) {
        unsigned char val = (unsigned char) (255 * input[blockIdx.z * width * height + width * row + col]);
    
        float temp = 255 * (cdf[val] - cdf[0]) / (1.0 - cdf[0]);
        float clamp = min(max(temp, 0.0), 255.0);
    
        output[blockIdx.z * width * height + width * row + col] = clamp / 255.0;
    }
}

int main() {
    // init image
    const int height = 9;
    const int width = 9;
    const int channels = 3;
    float imageHost[height * width * channels];

    // fill image array
    for (int i = 0; i < height * width * channels; i++)
        imageHost[i] = (float) rand() / RAND_MAX;

    // intermediate arrays
    float* imageDevice;
    unsigned char* greyscaleDevice;
    int* histogramDevice;
    float* CDFDevice;
    float* outputDevice;
    float* outputHost;

    // space allocation for intermediate arrays
    cudaMalloc((void**) &imageDevice, sizeof(float) * height * width * channels); 
    cudaMalloc((void**) &greyscaleDevice, sizeof(unsigned char) * height * width);
    cudaMalloc((void**) &histogramDevice, sizeof(int) * HISTOGRAM_LENGTH);
    cudaMalloc((void**) &CDFDevice, sizeof(float) * HISTOGRAM_LENGTH);

    cudaMalloc((void**) &outputDevice, sizeof(float) * height * width * channels);

    // assign data 
    cudaMemcpy(imageDevice, imageHost, sizeof(float) * height * width * channels, cudaMemcpyHostToDevice);
    cudaMemset(histogramDevice, 0, sizeof(int) * HISTOGRAM_LENGTH);

    // allocate memory for output
    outputHost = (float*) malloc(sizeof(float) * height * width * channels);

    // dim sizes
    dim3 blockDim;
    dim3 gridDim;

    // greyscale the image
    blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    gridDim = dim3(ceil(((float)width) / BLOCK_SIZE), ceil(((float)height / BLOCK_SIZE)), 1);
    greyscale<<<gridDim, blockDim>>>(imageDevice, greyscaleDevice, height, width, channels);
    cudaDeviceSynchronize();
    
    // trash histogram creation for benchmarking
    // blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    // gridDim = dim3(ceil(((float)width) / BLOCK_SIZE), ceil(((float)height / BLOCK_SIZE)), 1);
    // trash_histogram<<<gridDim, blockDim>>>(greyscaleDevice, histogramDevice, height, width);
    // cudaDeviceSynchronize();

    // create histogram
    blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    gridDim = dim3(ceil(((float)width) / BLOCK_SIZE), ceil(((float)height / BLOCK_SIZE)), 1);
    histogram<<<gridDim, blockDim>>>(greyscaleDevice, histogramDevice, height, width);
    cudaDeviceSynchronize();

    // create CDF from histogram
    blockDim = dim3(HISTOGRAM_LENGTH / 2);
    gridDim = dim3(1, 1, 1);
    CDF<<<gridDim, blockDim>>>(histogramDevice, CDFDevice, height, width);
    cudaDeviceSynchronize();

    // equalize image from CDF
    blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    gridDim = dim3(ceil(((float)width) / BLOCK_SIZE), ceil(((float)height / BLOCK_SIZE)), channels);
    equalization<<<gridDim, blockDim>>>(imageDevice, CDFDevice, outputDevice, height, width);
    cudaMemcpy(outputHost, outputDevice, sizeof(float) * height * width * channels, cudaMemcpyDeviceToHost);

    // DO NOT COPY OVER: for testing purposes only
    for (int i = 0; i < height * width * channels; i++) 
        std::cout << imageHost[i] * 255 << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < height * width * channels; i++) 
        std::cout << outputHost[i] << " ";
    std::cout << std::endl;

    // free memory on device and host
    cudaFree(imageDevice);
    cudaFree(greyscaleDevice);
    cudaFree(histogramDevice);
    cudaFree(CDFDevice);
    cudaFree(outputDevice);
    free(outputHost);

    // success 
    return 0;
}
