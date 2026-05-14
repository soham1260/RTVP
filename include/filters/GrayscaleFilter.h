#pragma once
#include "VideoFilter.h"

class GrayscaleFilter : public VideoFilter 
{
    public:
        GrayscaleFilter() {};
        ~GrayscaleFilter() {};

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};