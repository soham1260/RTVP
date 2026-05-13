#pragma once
#include "VideoFilter.h"

class ChromaKeyFilter : public VideoFilter 
{
    public:
        ChromaKeyFilter() {};
        ~ChromaKeyFilter() {};

        void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream);
};
