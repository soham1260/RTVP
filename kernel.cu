#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring> 

static cudaStream_t streams[3];
static unsigned char* d_fg[3];
static unsigned char* d_bg[3];
static unsigned char* h_pinned_fg[3];
static unsigned char* h_pinned_bg[3];
static int g_width, g_height, g_channels, g_size;

__global__ void processPixelKernel(unsigned char* d_image, unsigned char* d_bg, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;

        float b_raw = d_image[idx + 0] / 255.0f;
        float g_raw = d_image[idx + 1] / 255.0f;
        float r_raw = d_image[idx + 2] / 255.0f;

        float h, s, v;
        float max_val = fmaxf(r_raw, fmaxf(g_raw, b_raw));
        float min_val = fminf(r_raw, fminf(g_raw, b_raw));
        float delta = max_val - min_val;

        v = max_val;
        s = (max_val > 0.0f) ? (delta / max_val) : 0.0f;

        if (delta == 0) h = 0;
        else 
        {
            if (max_val == r_raw) h = 60.0f * (fmodf(((g_raw - b_raw) / delta), 6.0f));
            else if (max_val == g_raw) h = 60.0f * (((b_raw - r_raw) / delta) + 2.0f);
            else if (max_val == b_raw) h = 60.0f * (((r_raw - g_raw) / delta) + 4.0f);
        }
        if (h < 0) h += 360.0f;

        float targetHue = 120.0f;
        float dist = fabsf(h - targetHue);
        float innerLimit = 30.0f; 
        float outerLimit = 50.0f; 
        float alpha = 1.0f;

        if (dist < outerLimit && s > 0.3f && v > 0.3f) 
        {
            alpha = (dist - innerLimit) / (outerLimit - innerLimit);
            if (alpha < 0.0f) alpha = 0.0f;
            if (alpha > 1.0f) alpha = 1.0f;
        }

        d_image[idx + 0] = (unsigned char)(d_image[idx + 0] * alpha + d_bg[idx + 0] * (1.0f - alpha));
        d_image[idx + 1] = (unsigned char)(d_image[idx + 1] * alpha + d_bg[idx + 1] * (1.0f - alpha));
        d_image[idx + 2] = (unsigned char)(d_image[idx + 2] * alpha + d_bg[idx + 2] * (1.0f - alpha));
    }
}

extern "C" void initGreenScreenPipeline(int width, int height, int channels)
{
    g_width = width;
    g_height = height;
    g_channels = channels;
    g_size = width * height * channels * sizeof(unsigned char);

    for (int i = 0; i < 3; i++) 
    {
        cudaMalloc(&d_fg[i], g_size);
        cudaMalloc(&d_bg[i], g_size);
        cudaMallocHost(&h_pinned_fg[i], g_size);
        cudaMallocHost(&h_pinned_bg[i], g_size);
        cudaStreamCreate(&streams[i]);
    }
}

extern "C" void processFrameAsync(unsigned char* fg_data, unsigned char* bg_data, int streamIdx) 
{
    memcpy(h_pinned_fg[streamIdx], fg_data, g_size);
    memcpy(h_pinned_bg[streamIdx], bg_data, g_size);

    cudaMemcpyAsync(d_fg[streamIdx], h_pinned_fg[streamIdx], g_size, cudaMemcpyHostToDevice, streams[streamIdx]);
    cudaMemcpyAsync(d_bg[streamIdx], h_pinned_bg[streamIdx], g_size, cudaMemcpyHostToDevice, streams[streamIdx]);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((g_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (g_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    processPixelKernel<<<numBlocks, threadsPerBlock, 0, streams[streamIdx]>>>(d_fg[streamIdx], d_bg[streamIdx], g_width, g_height, g_channels);

    cudaMemcpyAsync(h_pinned_fg[streamIdx], d_fg[streamIdx], g_size, cudaMemcpyDeviceToHost, streams[streamIdx]);
}

extern "C" unsigned char* syncAndGetFrame(int streamIdx) 
{
    cudaStreamSynchronize(streams[streamIdx]);
    return h_pinned_fg[streamIdx];
}

extern "C" void cleanupGreenScreenPipeline()
{
    for (int i = 0; i < 3; i++) {
        cudaFree(d_fg[i]);
        cudaFree(d_bg[i]);
        cudaFreeHost(h_pinned_fg[i]); 
        cudaFreeHost(h_pinned_bg[i]);
        cudaStreamDestroy(streams[i]);
    }
}