#include "SimpleCV.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace SimpleCV
{
    static inline bool affine_inverse_2x3(const Matf &M, float Mi[6])
    {
        if (M.empty() || M.height != 2 || M.width != 3 || M.channels != 1)
            return false;

        const float a = M.at(0, 0), b = M.at(0, 1), tx = M.at(0, 2);
        const float c = M.at(1, 0), d = M.at(1, 1), ty = M.at(1, 2);

        const float det = a * d - b * c;
        if (std::fabs(det) < 1e-12f)
            return false;

        const float inv = 1.0f / det;
        Mi[0] = d * inv;
        Mi[1] = -b * inv;
        Mi[2] = (b * ty - d * tx) * inv;
        Mi[3] = -c * inv;
        Mi[4] = a * inv;
        Mi[5] = (c * tx - a * ty) * inv;
        return true;
    }

    // 取边界采样位置：CONSTANT 走 border value（由 caller 处理），其它（默认按 REPLICATE）夹紧
    static inline void clamp_xy(int &x, int &y, int w, int h)
    {
        if (x < 0) x = 0;
        else if (x >= w) x = w - 1;
        if (y < 0) y = 0;
        else if (y >= h) y = h - 1;
    }

    void warpAffine(const Mat &src, Mat &dst, const Matf &M, Size dsize,
                    BorderType borderType, const Scalar &borderValue)
    {
        if (src.empty() || dsize.width <= 0 || dsize.height <= 0)
        {
            dst.release();
            return;
        }
        if (src.depth != Depth::U8 || src.elem_size != 1)
        {
            dst.release();
            return;
        }
        const int c = src.channels;
        if (c < 1 || c > 4)
        {
            dst.release();
            return;
        }

        float Mi[6];
        if (!affine_inverse_2x3(M, Mi))
        {
            dst.release();
            return;
        }

        const int dh = dsize.height;
        const int dw = dsize.width;

        // dst 已就位且形状匹配则复用，否则重新分配；同源直接 in-place 不安全，分配新 buffer
        bool can_reuse =
            !dst.empty() &&
            dst.data != src.data &&
            dst.height == dh &&
            dst.width == dw &&
            dst.channels == c &&
            dst.depth == Depth::U8 &&
            dst.elem_size == 1 &&
            dst.step >= dw * c;
        if (!can_reuse)
            dst.create(dh, dw, c);

        unsigned char bv[4] = {0, 0, 0, 0};
        for (int k = 0; k < c; ++k)
            bv[k] = borderValue.v[k];

        const int sw = src.width;
        const int sh = src.height;
        const int sstep = src.step;
        const unsigned char *sdata = src.data;

        const float m00 = Mi[0], m01 = Mi[1], m02 = Mi[2];
        const float m10 = Mi[3], m11 = Mi[4], m12 = Mi[5];

        const bool use_constant = (borderType == BorderType::CONSTANT);

        for (int y = 0; y < dh; ++y)
        {
            unsigned char *drow = dst.data + (size_t)y * (size_t)dst.step;

            // 起点：x=0 时的 (sx, sy)
            float fx_row = m01 * y + m02;
            float fy_row = m11 * y + m12;

            for (int x = 0; x < dw; ++x)
            {
                const float fx = m00 * x + fx_row;
                const float fy = m10 * x + fy_row;

                const int x0 = (int)std::floor(fx);
                const int y0 = (int)std::floor(fy);
                const float ax = fx - (float)x0;
                const float ay = fy - (float)y0;

                const float w00 = (1.0f - ax) * (1.0f - ay);
                const float w10 = ax * (1.0f - ay);
                const float w01 = (1.0f - ax) * ay;
                const float w11 = ax * ay;

                unsigned char *dp = drow + (size_t)x * (size_t)c;

                // 完全在合法 2x2 邻域内：避免分支，最快路径
                if (x0 >= 0 && y0 >= 0 && x0 + 1 < sw && y0 + 1 < sh)
                {
                    const unsigned char *p00 = sdata + (size_t)y0 * sstep + (size_t)x0 * c;
                    const unsigned char *p10 = p00 + c;
                    const unsigned char *p01 = p00 + sstep;
                    const unsigned char *p11 = p01 + c;
                    for (int k = 0; k < c; ++k)
                    {
                        float v = p00[k] * w00 + p10[k] * w10 + p01[k] * w01 + p11[k] * w11;
                        int iv = (int)(v + 0.5f);
                        if (iv < 0) iv = 0;
                        else if (iv > 255) iv = 255;
                        dp[k] = (unsigned char)iv;
                    }
                    continue;
                }

                // 边界：每个邻居单独取（落在 src 内 / 越界用 border）
                unsigned char p00[4], p10[4], p01[4], p11[4];
                int xs[2] = {x0, x0 + 1};
                int ys[2] = {y0, y0 + 1};

                auto fetch = [&](int xi, int yi, unsigned char out[4]) {
                    if ((unsigned)xi < (unsigned)sw && (unsigned)yi < (unsigned)sh)
                    {
                        const unsigned char *sp = sdata + (size_t)yi * sstep + (size_t)xi * c;
                        for (int k = 0; k < c; ++k) out[k] = sp[k];
                    }
                    else if (use_constant)
                    {
                        for (int k = 0; k < c; ++k) out[k] = bv[k];
                    }
                    else
                    {
                        int cx = xi, cy = yi;
                        clamp_xy(cx, cy, sw, sh);
                        const unsigned char *sp = sdata + (size_t)cy * sstep + (size_t)cx * c;
                        for (int k = 0; k < c; ++k) out[k] = sp[k];
                    }
                };

                fetch(xs[0], ys[0], p00);
                fetch(xs[1], ys[0], p10);
                fetch(xs[0], ys[1], p01);
                fetch(xs[1], ys[1], p11);

                for (int k = 0; k < c; ++k)
                {
                    float v = p00[k] * w00 + p10[k] * w10 + p01[k] * w01 + p11[k] * w11;
                    int iv = (int)(v + 0.5f);
                    if (iv < 0) iv = 0;
                    else if (iv > 255) iv = 255;
                    dp[k] = (unsigned char)iv;
                }
            }
        }
    }
}

namespace SimpleCV
{
    // 求解 4x4 线性系统 A * x = b，A 列主元 Gauss-Jordan，结果写回 x
    // A 在过程中被破坏。成功返回 true。
    static bool solve_4x4(double A[4][4], double b[4], double x[4])
    {
        double M[4][5];
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j) M[i][j] = A[i][j];
            M[i][4] = b[i];
        }

        for (int i = 0; i < 4; ++i)
        {
            int piv = i;
            double mx = std::fabs(M[i][i]);
            for (int r = i + 1; r < 4; ++r)
            {
                double v = std::fabs(M[r][i]);
                if (v > mx) { mx = v; piv = r; }
            }
            if (mx < 1e-12)
                return false;

            if (piv != i)
            {
                for (int cc = 0; cc < 5; ++cc) std::swap(M[i][cc], M[piv][cc]);
            }

            const double diag = M[i][i];
            for (int cc = 0; cc < 5; ++cc) M[i][cc] /= diag;

            for (int r = 0; r < 4; ++r)
            {
                if (r == i) continue;
                const double f = M[r][i];
                if (f == 0.0) continue;
                for (int cc = 0; cc < 5; ++cc) M[r][cc] -= f * M[i][cc];
            }
        }

        for (int i = 0; i < 4; ++i) x[i] = M[i][4];
        return true;
    }

    // 4-DOF 相似变换最小二乘：对每对点 (x_i, y_i) -> (u_i, v_i) 建立 2 行
    //   [ x_i, -y_i, 1, 0 ] [a]   [u_i]
    //   [ y_i,  x_i, 0, 1 ] [b] = [v_i]
    //                       [tx]
    //                       [ty]
    // 解出 (a, b, tx, ty) 后构造仿射矩阵 [[a,-b,tx],[b,a,ty]]
    Matf estimateAffinePartial2D(const std::vector<Point2f> &from,
                                 const std::vector<Point2f> &to)
    {
        if (from.size() < 2 || from.size() != to.size())
            return Matf();

        double AtA[4][4] = {{0}};
        double Atb[4] = {0};

        const int N = (int)from.size();
        for (int i = 0; i < N; ++i)
        {
            const double xs = from[i].x, ys = from[i].y;
            const double u = to[i].x, v = to[i].y;

            // 对称累加 A^T A 和 A^T b（上三角，最后镜像）
            AtA[0][0] += xs * xs + ys * ys;
            AtA[1][1] += xs * xs + ys * ys;
            AtA[2][2] += 1.0;
            AtA[3][3] += 1.0;

            // (0,1): xs*(-ys) + ys*xs = 0 — 略
            AtA[0][2] += xs;
            AtA[0][3] += ys;
            AtA[1][2] += -ys;
            AtA[1][3] += xs;
            // (2,3): 1*0 + 0*1 = 0 — 略

            Atb[0] += xs * u + ys * v;
            Atb[1] += -ys * u + xs * v;
            Atb[2] += u;
            Atb[3] += v;
        }
        // 镜像下三角
        AtA[1][0] = AtA[0][1];
        AtA[2][0] = AtA[0][2];
        AtA[3][0] = AtA[0][3];
        AtA[2][1] = AtA[1][2];
        AtA[3][1] = AtA[1][3];
        AtA[3][2] = AtA[2][3];

        double sol[4];
        if (!solve_4x4(AtA, Atb, sol))
            return Matf();

        const float a = (float)sol[0];
        const float b = (float)sol[1];
        const float tx = (float)sol[2];
        const float ty = (float)sol[3];

        Matf out(2, 3);
        out.at(0, 0) = a;  out.at(0, 1) = -b; out.at(0, 2) = tx;
        out.at(1, 0) = b;  out.at(1, 1) = a;  out.at(1, 2) = ty;
        return out;
    }
}
