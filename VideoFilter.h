#pragma once

class VideoFilter
{
    public:
        virtual ~VideoFilter() {};
        
        virtual void process(unsigned char* d_fg, unsigned char* d_bg, int width, int height, int channels, cudaStream_t stream) = 0;
};
