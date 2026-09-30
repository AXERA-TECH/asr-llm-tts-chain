#pragma once
#include "SimpleCV.hpp"

namespace SimpleCV
{
    static inline ColorSpace infer_space_from_channels(const Mat &m)
    {
        if (m.channels == 1)
            return ColorSpace::GRAY;
        if (m.channels == 3)
            return ColorSpace::RGB; // 默认按 RGB 处理
        if (m.channels == 4)
            return ColorSpace::RGBA; // 默认按 RGBA 处理
        return ColorSpace::UNCHANGED;
    }

    static inline int desired_channels(ColorSpace flag)
    {
        switch (flag)
        {
        case ColorSpace::GRAY:
            return 1;
        case ColorSpace::RGB:
        case ColorSpace::BGR:
            return 3;
        case ColorSpace::RGBA:
        case ColorSpace::BGRA:
            return 4;
        case ColorSpace::UNCHANGED:
        default:
            return 0; // stb: 0 = keep original
        }
    }

    static inline bool is_bgr_family(ColorSpace s)
    {
        return s == ColorSpace::BGR || s == ColorSpace::BGRA;
    }

    static inline bool is_rgb_family(ColorSpace s)
    {
        return s == ColorSpace::RGB || s == ColorSpace::RGBA;
    }

    // 交换 R<->B（适用于 3/4 通道）
    static inline void swap_rb_inplace(Mat &m)
    {
        if (m.empty())
            return;
        if (m.channels != 3 && m.channels != 4)
            return;

        for (int y = 0; y < m.height; ++y)
        {
            unsigned char *row = m.data + y * m.step;
            if (m.channels == 3)
            {
                for (int x = 0; x < m.width; ++x)
                {
                    unsigned char *p = row + x * 3;
                    std::swap(p[0], p[2]);
                }
            }
            else
            {
                for (int x = 0; x < m.width; ++x)
                {
                    unsigned char *p = row + x * 4;
                    std::swap(p[0], p[2]);
                }
            }
        }
    }
}
