#include "SimpleCV.hpp"
#include <algorithm>
#include <cstdint>
#include <cmath>

// 你已有的 namespace
namespace SimpleCV
{

    static inline bool in_bounds(const Mat &img, int x, int y)
    {
        return (unsigned)x < (unsigned)img.width && (unsigned)y < (unsigned)img.height;
    }

    static inline void put_pixel(Mat &img, int x, int y, const Scalar &color)
    {
        if (!in_bounds(img, x, y) || img.data == nullptr)
            return;

        unsigned char *p = img.data + (size_t)y * (size_t)img.step + (size_t)x * (size_t)img.channels;

        if (img.channels == 1)
        {
            p[0] = color.v[0];
        }
        else if (img.channels == 3)
        {
            p[0] = color.v[0];
            p[1] = color.v[1];
            p[2] = color.v[2];
        }
        else if (img.channels == 4)
        {
            p[0] = color.v[0];
            p[1] = color.v[1];
            p[2] = color.v[2];
            p[3] = color.v[3];
        }
    }

    static inline void hline(Mat &img, int x0, int x1, int y, const Scalar &color)
    {
        if ((unsigned)y >= (unsigned)img.height)
            return;
        if (x0 > x1)
            std::swap(x0, x1);
        x0 = std::max(0, x0);
        x1 = std::min(img.width - 1, x1);
        for (int x = x0; x <= x1; ++x)
            put_pixel(img, x, y, color);
    }

    static inline void vline(Mat &img, int x, int y0, int y1, const Scalar &color)
    {
        if ((unsigned)x >= (unsigned)img.width)
            return;
        if (y0 > y1)
            std::swap(y0, y1);
        y0 = std::max(0, y0);
        y1 = std::min(img.height - 1, y1);
        for (int y = y0; y <= y1; ++y)
            put_pixel(img, x, y, color);
    }

    // 简单的“画小圆点当作粗线”的方式（足够实用，易维护）
    static void circle_filled(Mat &img, Point center, int radius, const Scalar &color);

    static void line_thin(Mat &img, Point p0, Point p1, const Scalar &color)
    {
        int x0 = p0.x, y0 = p0.y;
        int x1 = p1.x, y1 = p1.y;

        int dx = std::abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
        int err = dx + dy;

        while (true)
        {
            put_pixel(img, x0, y0, color);
            if (x0 == x1 && y0 == y1)
                break;
            int e2 = err << 1;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }

    SIMPLECV_API void line(Mat &img, Point p0, Point p1, const Scalar &color, int thickness)
    {
        if (img.empty() || thickness == 0)
            return;
        if (img.depth != Depth::U8 || img.elem_size != 1)
            return;

        if (thickness <= 1)
        {
            line_thin(img, p0, p1, color);
            return;
        }

        // 粗线：沿着细线每个点画一个填充圆（半径=thickness/2）
        int r = std::max(1, thickness / 2);
        int x0 = p0.x, y0 = p0.y;
        int x1 = p1.x, y1 = p1.y;

        int dx = std::abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
        int dy = -std::abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
        int err = dx + dy;

        while (true)
        {
            circle_filled(img, Point(x0, y0), r, color);
            if (x0 == x1 && y0 == y1)
                break;
            int e2 = err << 1;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }

    // 画圆：填充（用 scanline）
    static void circle_filled(Mat &img, Point c, int radius, const Scalar &color)
    {
        if (img.empty() || radius <= 0)
            return;

        int x0 = c.x, y0 = c.y;
        int r = radius;

        // Midpoint circle + scanline fill
        int x = r;
        int y = 0;
        int err = 1 - x;

        while (x >= y)
        {
            hline(img, x0 - x, x0 + x, y0 + y, color);
            hline(img, x0 - x, x0 + x, y0 - y, color);
            hline(img, x0 - y, x0 + y, y0 + x, color);
            hline(img, x0 - y, x0 + y, y0 - x, color);

            y++;
            if (err < 0)
            {
                err += 2 * y + 1;
            }
            else
            {
                x--;
                err += 2 * (y - x) + 1;
            }
        }
    }

    // 画圆：描边（thin）
    static void circle_outline(Mat &img, Point c, int radius, const Scalar &color)
    {
        if (img.empty() || radius <= 0)
            return;

        int x0 = c.x, y0 = c.y;
        int x = radius;
        int y = 0;
        int err = 1 - x;

        while (x >= y)
        {
            put_pixel(img, x0 + x, y0 + y, color);
            put_pixel(img, x0 + y, y0 + x, color);
            put_pixel(img, x0 - y, y0 + x, color);
            put_pixel(img, x0 - x, y0 + y, color);
            put_pixel(img, x0 - x, y0 - y, color);
            put_pixel(img, x0 - y, y0 - x, color);
            put_pixel(img, x0 + y, y0 - x, color);
            put_pixel(img, x0 + x, y0 - y, color);

            y++;
            if (err < 0)
            {
                err += 2 * y + 1;
            }
            else
            {
                x--;
                err += 2 * (y - x) + 1;
            }
        }
    }

    SIMPLECV_API void circle(Mat &img, Point center, int radius, const Scalar &color, int thickness)
    {
        if (img.empty() || radius <= 0 || thickness == 0)
            return;
        if (img.depth != Depth::U8 || img.elem_size != 1)
            return;

        if (thickness < 0)
        {
            circle_filled(img, center, radius, color);
            return;
        }

        if (thickness == 1)
        {
            circle_outline(img, center, radius, color);
            return;
        }

        // 厚圆：画多个半径的 outline
        int half = thickness / 2;
        int r0 = std::max(1, radius - half);
        int r1 = radius + (thickness - half - 1);
        for (int r = r0; r <= r1; ++r)
            circle_outline(img, center, r, color);
    }

    SIMPLECV_API void rectangle(Mat &img, Point pt1, Point pt2, const Scalar &color,
                                int thickness, int /*lineType*/, int /*shift*/)
    {
        if (img.empty() || thickness == 0)
            return;
        if (img.depth != Depth::U8 || img.elem_size != 1)
            return;

        int x0 = std::min(pt1.x, pt2.x);
        int x1 = std::max(pt1.x, pt2.x);
        int y0 = std::min(pt1.y, pt2.y);
        int y1 = std::max(pt1.y, pt2.y);

        if (thickness < 0)
        {
            // filled
            y0 = std::max(0, y0);
            y1 = std::min(img.height - 1, y1);
            for (int y = y0; y <= y1; ++y)
                hline(img, x0, x1, y, color);
            return;
        }

        int t = std::max(1, thickness);

        // top & bottom bands
        for (int k = 0; k < t; ++k)
        {
            hline(img, x0, x1, y0 + k, color);
            hline(img, x0, x1, y1 - k, color);
        }
        // left & right bands
        for (int k = 0; k < t; ++k)
        {
            vline(img, x0 + k, y0, y1, color);
            vline(img, x1 - k, y0, y1, color);
        }
    }

    SIMPLECV_API void rectangle(Mat &img, Rect rec, const Scalar &color, int thickness)
    {
        Point p1(rec.x, rec.y);
        Point p2(rec.x + rec.width, rec.y + rec.height);
        // OpenCV 的 Rect 右下角是 (x+width, y+height) 不包含边界；
        // 这里画图更直观，减 1 让它包含边界
        p2.x -= 1;
        p2.y -= 1;
        rectangle(img, p1, p2, color, thickness, 8, 0);
    }

} // namespace SimpleCV
