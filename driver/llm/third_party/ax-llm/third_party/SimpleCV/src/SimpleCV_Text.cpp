#include "SimpleCV.hpp"
#include <algorithm>
#include <cstdint>
#include <cmath>

#include "mat_pixel_drawing_font.h"

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace SimpleCV
{

    Size getTextSize(const std::string &text,
                     double fontScale,
                     int thickness,
                     int *baseLine)
    {
        int scale = (int)std::lround(std::max(1.0, fontScale));
        int t = std::max(1, thickness);

        const int gw = 20 * scale;
        const int gh = 40 * scale;
        const int gap = 2 * scale; // 字间距
        const int line_gap = 6 * scale;

        int w = 0, line_w = 0, lines = 1;
        for (char c : text)
        {
            if (c == '\n')
            {
                w = std::max(w, line_w);
                line_w = 0;
                ++lines;
            }
            else
            {
                line_w += gw + gap;
            }
        }
        w = std::max(w, line_w);
        if (w > 0)
            w -= gap; // 最后一个字符不加 gap

        int h = lines * gh + (lines - 1) * line_gap;

        // baseline：按经验给“字形底部往上 6 像素”
        if (baseLine)
            *baseLine = 6 * scale + (t - 1);

        // thickness 让外轮廓变粗，尺寸稍微增大
        w += (t - 1) * 2;
        h += (t - 1) * 2;

        return Size(w, h);
    }

       static inline void blend_pixel(Mat &img, int x, int y,
                                   const Scalar &color,
                                   unsigned char a /*0..255*/)
    {
        if (!img.data)
            return;
        if ((unsigned)x >= (unsigned)img.width || (unsigned)y >= (unsigned)img.height)
            return;
        if (a == 0)
            return;

        unsigned char *p = img.data + (size_t)y * (size_t)img.step + (size_t)x * (size_t)img.channels;

        // alpha blend: dst = dst*(1-a) + src*a
        // 用整数： (dst*(255-a) + src*a + 127)/255
        const int ia = a;
        const int inv = 255 - ia;

        if (img.channels == 1)
        {
            int dst0 = p[0];
            int src0 = color.v[0];
            p[0] = (unsigned char)((dst0 * inv + src0 * ia + 127) / 255);
        }
        else if (img.channels == 3)
        {
            for (int c = 0; c < 3; ++c)
            {
                int dstc = p[c];
                int srcc = color.v[c];
                p[c] = (unsigned char)((dstc * inv + srcc * ia + 127) / 255);
            }
        }
        else if (img.channels == 4)
        {
            // RGB 做 blend，alpha 通道你可以选择：保持不变 / 也做 blend / 直接写 255
            for (int c = 0; c < 3; ++c)
            {
                int dstc = p[c];
                int srcc = color.v[c];
                p[c] = (unsigned char)((dstc * inv + srcc * ia + 127) / 255);
            }
            // 可选：让 alpha 更“实”
            // p[3] = std::max(p[3], a);
        }
    }

    static void draw_glyph_20x40_gray(Mat &img, int x0, int y0,
                                      const unsigned char *glyph /*20*40*/,
                                      int scale,
                                      const Scalar &color,
                                      int thickness)
    {
        const int GW = 20, GH = 40;
        int t = std::max(1, thickness);

        // thickness：用简单“膨胀”方式，把每个像素扩成 t×t 的块（再乘 scale）
        // 注意：字库是灰度，扩张也应该扩张 alpha
        for (int gy = 0; gy < GH; ++gy)
        {
            for (int gx = 0; gx < GW; ++gx)
            {
                unsigned char a = glyph[gy * GW + gx];
                if (a == 0)
                    continue;

                for (int oy = 0; oy < t; ++oy)
                {
                    for (int ox = 0; ox < t; ++ox)
                    {
                        int px = x0 + (gx * scale) + ox;
                        int py = y0 + (gy * scale) + oy;

                        // scale: 画 scale×scale block
                        for (int dy = 0; dy < scale; ++dy)
                            for (int dx = 0; dx < scale; ++dx)
                                blend_pixel(img, px + dx, py + dy, color, a);
                    }
                }
            }
        }
    }

    SIMPLECV_API void putText(Mat &img,
                              const std::string &text,
                              Point org,
                              double fontScale,
                              const Scalar &color,
                              int thickness,
                              int /*lineType*/,
                              bool bottomLeftOrigin)
    {
        if (img.empty() || !img.data || text.empty())
            return;
        if (img.depth != Depth::U8 || img.elem_size != 1)
            return;

        int scale = (int)std::lround(std::max(1.0, fontScale));
        int t = std::max(1, thickness);

        const int gw = 20 * scale;
        const int gh = 40 * scale;
        const int gap = 2 * scale;
        const int line_gap = 6 * scale;

        // 你希望 org 是什么语义？
        // 我这里按 OpenCV：org 是 baseline-left
        // glyph top = baseline - gh
        int x = org.x;
        int baseline_y = org.y;

        for (size_t i = 0; i < text.size(); ++i)
        {
            unsigned char ch = (unsigned char)text[i];
            if (ch == '\n')
            {
                x = org.x;
                baseline_y += gh + line_gap;
                continue;
            }

            if (ch < 0x20 || ch > 0x7E)
            {
                // 不支持的字符用 '?' 代替
                ch = '?';
            }
            int idx = (int)ch - 0x20;

            int y_top = baseline_y - gh;
            if (bottomLeftOrigin)
            {
                // 把 baseline_y 也映射到内部 top-left 坐标
                // baseline_y 是以左下为原点的 y，转为左上原点：
                int baseline_y_tl = (img.height - 1) - baseline_y;
                y_top = baseline_y_tl - gh + 1;
            }

            draw_glyph_20x40_gray(img, x, y_top, mono_font_data[idx], scale, color, t);

            x += gw + gap;
        }
    }

} // namespace SimpleCV
