#include "SimpleCV.hpp"
#include "SimpleCV_Common.hpp"

#ifndef STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif

#include "stb_image_resize2.h"

namespace SimpleCV
{
    void resize(const Mat &src, Mat &dst, int dst_width, int dst_height)
    {
        if (src.empty() || dst_width <= 0 || dst_height <= 0)
        {
            dst.release();
            return;
        }
        if (src.depth != Depth::U8 || src.elem_size != 1)
        {
            dst.release();
            return;
        }
        if (src.step < src.width * src.channels * src.elem_size)
        { // 防御：src stride 必须够
            dst.release();
            return;
        }

        // 处理 in-place / 共享内存（最简单：指针相等就当冲突）
        if (dst.data == src.data)
        {
            Mat tmp(dst_height, dst_width, src.channels);
            resize(src, tmp, dst_width, dst_height);
            dst = tmp; // 或 swap
            return;
        }

        bool can_reuse =
            dst.data &&
            dst.width == dst_width &&
            dst.height == dst_height &&
            dst.channels == src.channels &&
            dst.depth == src.depth &&
            dst.elem_size == src.elem_size &&
            dst.step >= dst_width * dst.channels * dst.elem_size;

        if (!can_reuse)
            dst.create(dst_height, dst_width, src.channels, src.depth, 0);

        if (dst.step < dst.width * dst.channels * dst.elem_size)
        { // 防御：dst stride 必须够
            dst.release();
            return;
        }

        stbir_pixel_layout layout;
        switch (src.channels)
        {
        case 1:
            layout = STBIR_1CHANNEL;
            break;
        case 2:
            layout = STBIR_2CHANNEL;
            break;
        case 3:
            layout = STBIR_RGB;
            break;
        case 4:
            layout = STBIR_RGBA;
            break;
        default:
            dst.release();
            return;
        }

        unsigned char *out = stbir_resize_uint8_linear(
            src.data, src.width, src.height, src.step,
            dst.data, dst.width, dst.height, dst.step,
            layout);

        if (!out)
            dst.release();
    }

}

namespace SimpleCV
{
    static inline unsigned char clamp_u8(int v)
    {
        return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }

    static inline unsigned char rgb_to_gray_u8(unsigned char r, unsigned char g, unsigned char b)
    {
        // 0.299R + 0.587G + 0.114B
        int y = (299 * (int)r + 587 * (int)g + 114 * (int)b + 500) / 1000;
        return clamp_u8(y);
    }

    static inline bool dst_buffer_compatible(const Mat &dst, int h, int w, int c)
    {
        if (dst.empty())
            return false;
        if (dst.depth != Depth::U8 || dst.elem_size != 1)
            return false;
        if (dst.height != h || dst.width != w || dst.channels != c)
            return false;
        const int min_step = w * c * dst.elem_size;
        return dst.step >= min_step && dst.data != nullptr;
    }

    void cvtColor(const Mat &src, Mat &dst, ColorSpace dst_space, ColorSpace src_space)
    {
        if (src.empty())
        {
            dst.release();
            return;
        }
        if (src.depth != Depth::U8 || src.elem_size != 1)
        {
            dst.release();
            return;
        }

        if (src_space == ColorSpace::AUTO)
            src_space = infer_space_from_channels(src);

        // 先把 src_space 归一到 {GRAY, RGB, BGR, RGBA, BGRA}
        if (src_space == ColorSpace::UNCHANGED)
            src_space = infer_space_from_channels(src);

        // dst_space 也不允许 UNCHANGED/AUTO（你也可以允许 AUTO=按 src 直接返回 clone）
        if (dst_space == ColorSpace::AUTO || dst_space == ColorSpace::UNCHANGED)
        {
            // AUTO/UNCHANGED：保持源格式，做浅拷贝（共享底层 buffer）
            dst = src;
            return;
        }

        auto dst_ch = desired_channels(dst_space);
        if (dst_ch == 0)
        {
            dst.release();
            return;
        }

        // 如果用户提供的 dst 尺寸/通道/stride 满足需求，则直接复用；否则重新分配
        if (!dst_buffer_compatible(dst, src.height, src.width, dst_ch))
        {
            dst.create(src.height, src.width, dst_ch, src.depth, 0);
        }

        // 一些快速路径
        if (src_space == dst_space)
        {
            // channels 必须一致才能直接 copy
            if (src.channels == dst.channels && src.step == dst.step)
            {
                std::memcpy(dst.data, src.data, (size_t)src.height * (size_t)src.step);
                return;
            }
        }

        // 逐像素转换
        for (int y = 0; y < src.height; ++y)
        {
            const unsigned char *sp = src.data + y * src.step;
            unsigned char *dp = dst.data + y * dst.step;

            for (int x = 0; x < src.width; ++x)
            {
                // 先从 src 读出 (r,g,b,a) 的“逻辑值”
                unsigned char r = 0, g = 0, b = 0, a = 255;

                if (src_space == ColorSpace::GRAY)
                {
                    unsigned char v = sp[x];
                    r = g = b = v;
                    a = 255;
                }
                else if (src_space == ColorSpace::RGB)
                {
                    const unsigned char *p = sp + x * 3;
                    r = p[0];
                    g = p[1];
                    b = p[2];
                    a = 255;
                }
                else if (src_space == ColorSpace::BGR)
                {
                    const unsigned char *p = sp + x * 3;
                    b = p[0];
                    g = p[1];
                    r = p[2];
                    a = 255;
                }
                else if (src_space == ColorSpace::RGBA)
                {
                    const unsigned char *p = sp + x * 4;
                    r = p[0];
                    g = p[1];
                    b = p[2];
                    a = p[3];
                }
                else if (src_space == ColorSpace::BGRA)
                {
                    const unsigned char *p = sp + x * 4;
                    b = p[0];
                    g = p[1];
                    r = p[2];
                    a = p[3];
                }
                else
                {
                    // 不支持的 src_space
                    dst.release();
                    return;
                }

                // 写入 dst
                if (dst_space == ColorSpace::GRAY)
                {
                    dp[x] = rgb_to_gray_u8(r, g, b);
                }
                else if (dst_space == ColorSpace::RGB)
                {
                    unsigned char *q = dp + x * 3;
                    q[0] = r;
                    q[1] = g;
                    q[2] = b;
                }
                else if (dst_space == ColorSpace::BGR)
                {
                    unsigned char *q = dp + x * 3;
                    q[0] = b;
                    q[1] = g;
                    q[2] = r;
                }
                else if (dst_space == ColorSpace::RGBA)
                {
                    unsigned char *q = dp + x * 4;
                    q[0] = r;
                    q[1] = g;
                    q[2] = b;
                    q[3] = a;
                }
                else if (dst_space == ColorSpace::BGRA)
                {
                    unsigned char *q = dp + x * 4;
                    q[0] = b;
                    q[1] = g;
                    q[2] = r;
                    q[3] = a;
                }
                else
                {
                    dst.release();
                    return;
                }
            }
        }
    }

    Mat cvtColor(const Mat &src, ColorSpace dst_space, ColorSpace src_space)
    {
        Mat out;
        cvtColor(src, out, dst_space, src_space);
        return out;
    }
}

namespace SimpleCV
{
    static inline unsigned char border_pick_value(
        const std::vector<unsigned char> &v, int k)
    {
        if (v.empty())
            return 0;
        if (v.size() == 1)
            return v[0];
        if (v.size() == 3)
            return (k < 3) ? v[k] : 255;
        if (k < (int)v.size())
            return v[k];
        return v.back();
    }

    static inline int border_map_coord(int p, int len, BorderType bt)
    {
        // 把越界坐标 p 映射到 [0, len-1]
        // len 必须 > 0
        if (len == 1)
            return 0;

        if (bt == BorderType::REPLICATE)
        {
            if (p < 0)
                return 0;
            if (p >= len)
                return len - 1;
            return p;
        }

        if (bt == BorderType::REFLECT || bt == BorderType::REFLECT_101)
        {
            // OpenCV 语义：
            // REFLECT:      fedcba|abcdefgh|hgfedcb
            // REFLECT_101:   gfedcb|abcdefgh|gfedcba  （边界像素不重复）
            const int delta = (bt == BorderType::REFLECT_101) ? 1 : 0;

            while (p < 0 || p >= len)
            {
                if (p < 0)
                    p = -p - 1 + delta;
                else
                    p = (2 * len - 1) - p - delta;
            }
            // 保险
            if (p < 0)
                p = 0;
            if (p >= len)
                p = len - 1;
            return p;
        }

        // CONSTANT 不需要映射（调用方会走填充分支）
        // 为了安全：夹紧
        if (p < 0)
            return 0;
        if (p >= len)
            return len - 1;
        return p;
    }

    void copyMakeBorder(
        const Mat &src,
        Mat &dst,
        int top, int bottom, int left, int right,
        BorderType borderType,
        const std::vector<unsigned char> &value)
    {
        if (src.empty())
        {
            dst.release();
            return;
        }
        if (src.depth != Depth::U8 || src.elem_size != 1)
        {
            dst.release();
            return;
        }
        if (top < 0)
            top = 0;
        if (bottom < 0)
            bottom = 0;
        if (left < 0)
            left = 0;
        if (right < 0)
            right = 0;

        const int c = src.channels;
        const int out_h = src.height + top + bottom;
        const int out_w = src.width + left + right;

        if (out_h <= 0 || out_w <= 0 || c <= 0)
        {
            dst.release();
            return;
        }

        Mat out(out_h, out_w, c); // step = out_w*c (紧密)

        // ===== 1) CONSTANT：先整张填充，再把 src 贴进去（最快）=====
        if (borderType == BorderType::CONSTANT)
        {
            // fill
            for (int y = 0; y < out.height; ++y)
            {
                unsigned char *row = out.data + y * out.step;
                for (int x = 0; x < out.width; ++x)
                {
                    unsigned char *p = row + x * c;
                    for (int k = 0; k < c; ++k)
                        p[k] = border_pick_value(value, k);
                }
            }

            // paste center
            for (int y = 0; y < src.height; ++y)
            {
                unsigned char *drow = out.data + (y + top) * out.step + left * c;
                const unsigned char *srow = src.data + y * src.step;
                std::memcpy(drow, srow, (size_t)src.width * (size_t)c * (size_t)src.elem_size);
            }

            dst = std::move(out);
            return;
        }

        // ===== 2) 非 CONSTANT：逐像素用映射规则取 src 对应像素（简单通用）=====
        for (int y = 0; y < out.height; ++y)
        {
            const int sy = border_map_coord(y - top, src.height, borderType);
            const unsigned char *srow = src.data + sy * src.step;
            unsigned char *drow = out.data + y * out.step;

            for (int x = 0; x < out.width; ++x)
            {
                const int sx = border_map_coord(x - left, src.width, borderType);
                const unsigned char *sp = srow + sx * c;
                unsigned char *dp = drow + x * c;

                // copy pixel
                for (int k = 0; k < c; ++k)
                    dp[k] = sp[k];
            }
        }

        dst = std::move(out);
    }
}
