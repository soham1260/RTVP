#pragma once
#include "VideoFilter.h"

class SepiaFilter : public VideoFilter 
{
    public:
        SepiaFilter() {};
        ~SepiaFilter() {};

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
