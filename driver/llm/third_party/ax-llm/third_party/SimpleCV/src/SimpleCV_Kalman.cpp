#include "SimpleCV.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace SimpleCV
{
    static inline bool same_shape(const Matf &a, const Matf &b)
    {
        return !a.empty() && !b.empty() &&
               a.channels == 1 && b.channels == 1 &&
               a.height == b.height && a.width == b.width;
    }

    static Matf mat_add(const Matf &a, const Matf &b)
    {
        if (!same_shape(a, b))
            return Matf();
        Matf out(a.height, a.width);
        for (int r = 0; r < a.height; ++r)
        {
            const float *pa = a.ptr(r);
            const float *pb = b.ptr(r);
            float *po = out.ptr(r);
            for (int c = 0; c < a.width; ++c)
                po[c] = pa[c] + pb[c];
        }
        return out;
    }

    static Matf mat_sub(const Matf &a, const Matf &b)
    {
        if (!same_shape(a, b))
            return Matf();
        Matf out(a.height, a.width);
        for (int r = 0; r < a.height; ++r)
        {
            const float *pa = a.ptr(r);
            const float *pb = b.ptr(r);
            float *po = out.ptr(r);
            for (int c = 0; c < a.width; ++c)
                po[c] = pa[c] - pb[c];
        }
        return out;
    }

    static Matf mat_t(const Matf &a)
    {
        if (a.empty())
            return Matf();
        if (a.channels != 1)
            return Matf();
        Matf out(a.width, a.height);
        for (int r = 0; r < a.height; ++r)
            for (int c = 0; c < a.width; ++c)
                out.at(c, r) = a.at(r, c);
        return out;
    }

    static Matf mat_mul(const Matf &a, const Matf &b)
    {
        if (a.empty() || b.empty() || a.channels != 1 || b.channels != 1 || a.width != b.height)
            return Matf();

        Matf out(a.height, b.width);
        out.setZero();

        // i-k-j loop: better cache locality for row-major a/out
        for (int i = 0; i < a.height; ++i)
        {
            const float *arow = a.ptr(i);
            float *orow = out.ptr(i);

            for (int k = 0; k < a.width; ++k)
            {
                const float aik = arow[k];
                if (aik == 0.0f)
                    continue;

                const float *brow = b.ptr(k);
                for (int j = 0; j < b.width; ++j)
                    orow[j] += aik * brow[j];
            }
        }

        return out;
    }

    static bool mat_inv(const Matf &src, Matf &dst)
    {
        if (src.empty() || src.channels != 1 || src.height != src.width)
        {
            dst.release();
            return false;
        }

        const int n = src.height;
        Matf a = src.clone(); // 深拷贝：后面会改写 a
        dst = Matf::eye(n);

        const float eps = 1e-8f;

        for (int i = 0; i < n; ++i)
        {
            // pivot: max abs in column i from row i..n-1
            int pivot = i;
            float max_abs = std::fabs(a.at(i, i));
            for (int r = i + 1; r < n; ++r)
            {
                float v = std::fabs(a.at(r, i));
                if (v > max_abs)
                {
                    max_abs = v;
                    pivot = r;
                }
            }

            if (max_abs < eps)
            {
                dst.release();
                return false;
            }

            if (pivot != i)
            {
                for (int c = 0; c < n; ++c)
                {
                    std::swap(a.at(i, c), a.at(pivot, c));
                    std::swap(dst.at(i, c), dst.at(pivot, c));
                }
            }

            const float diag = a.at(i, i);
            for (int c = 0; c < n; ++c)
            {
                a.at(i, c) /= diag;
                dst.at(i, c) /= diag;
            }

            for (int r = 0; r < n; ++r)
            {
                if (r == i)
                    continue;
                const float f = a.at(r, i);
                if (f == 0.0f)
                    continue;
                for (int c = 0; c < n; ++c)
                {
                    a.at(r, c) -= f * a.at(i, c);
                    dst.at(r, c) -= f * dst.at(i, c);
                }
            }
        }

        return true;
    }
}

namespace SimpleCV
{
    void KalmanFilter::init(int dynam_params, int measure_params, int control_params)
    {
        dynamParams = dynam_params;
        measureParams = measure_params;
        controlParams = control_params;

        if (dynamParams <= 0 || measureParams <= 0 || controlParams < 0)
        {
            *this = KalmanFilter();
            return;
        }

        statePre.create(dynamParams, 1);
        statePre.setZero();
        statePost.create(dynamParams, 1);
        statePost.setZero();

        transitionMatrix = Matf::eye(dynamParams);

        if (controlParams > 0)
        {
            controlMatrix.create(dynamParams, controlParams);
            controlMatrix.setZero();
        }
        else
        {
            controlMatrix.release();
        }

        measurementMatrix.create(measureParams, dynamParams);
        measurementMatrix.setZero();
        for (int i = 0; i < std::min(measureParams, dynamParams); ++i)
            measurementMatrix.at(i, i) = 1.0f;

        processNoiseCov = Matf::eye(dynamParams);
        measurementNoiseCov = Matf::eye(measureParams);

        errorCovPre = Matf::eye(dynamParams);
        errorCovPost = Matf::eye(dynamParams);

        gain.create(dynamParams, measureParams);
        gain.setZero();
    }

    const Matf &KalmanFilter::predict(const Matf &control)
    {
        if (dynamParams <= 0 || measureParams <= 0)
            return statePre;

        const bool shape_ok =
            transitionMatrix.channels == 1 &&
            transitionMatrix.rows() == dynamParams &&
            transitionMatrix.cols() == dynamParams &&
            statePost.channels == 1 &&
            statePost.rows() == dynamParams &&
            statePost.cols() == 1 &&
            errorCovPost.channels == 1 &&
            errorCovPost.rows() == dynamParams &&
            errorCovPost.cols() == dynamParams;

        if (!shape_ok)
        {
            statePre = statePost;
            errorCovPre = errorCovPost;
            return statePre;
        }

        // x'(k) = A x(k-1) + B u(k)
        statePre = mat_mul(transitionMatrix, statePost);
        if (controlParams > 0 &&
            controlMatrix.channels == 1 &&
            controlMatrix.rows() == dynamParams &&
            controlMatrix.cols() == controlParams &&
            !control.empty() &&
            control.channels == 1 &&
            control.rows() == controlParams &&
            control.cols() == 1)
        {
            Matf Bu = mat_mul(controlMatrix, control);
            if (!Bu.empty())
                statePre = mat_add(statePre, Bu);
        }

        // P'(k) = A P(k-1) A^T + Q
        Matf At = mat_t(transitionMatrix);
        Matf AP = mat_mul(transitionMatrix, errorCovPost);
        Matf APA = mat_mul(AP, At);
        if (!processNoiseCov.empty() &&
            processNoiseCov.channels == 1 &&
            processNoiseCov.rows() == dynamParams &&
            processNoiseCov.cols() == dynamParams)
        {
            errorCovPre = mat_add(APA, processNoiseCov);
        }
        else
        {
            errorCovPre = APA;
        }

        return statePre;
    }

    const Matf &KalmanFilter::correct(const Matf &measurement)
    {
        if (dynamParams <= 0 || measureParams <= 0)
            return statePost;

        if (statePre.empty())
        {
            statePre = statePost;
            errorCovPre = errorCovPost;
        }

        const bool shape_ok =
            measurementMatrix.channels == 1 &&
            measurementMatrix.rows() == measureParams &&
            measurementMatrix.cols() == dynamParams &&
            errorCovPre.channels == 1 &&
            errorCovPre.rows() == dynamParams &&
            errorCovPre.cols() == dynamParams &&
            !measurement.empty() &&
            measurement.channels == 1 &&
            measurement.rows() == measureParams &&
            measurement.cols() == 1;

        if (!shape_ok)
        {
            statePost = statePre;
            errorCovPost = errorCovPre;
            return statePost;
        }

        // y = z - H x'
        Matf Hx = mat_mul(measurementMatrix, statePre);
        Matf y = mat_sub(measurement, Hx);

        // S = H P' H^T + R
        Matf Ht = mat_t(measurementMatrix);
        Matf PHt = mat_mul(errorCovPre, Ht);                // (d x m)
        Matf S = mat_mul(measurementMatrix, PHt);           // (m x m)
        if (!measurementNoiseCov.empty() &&
            measurementNoiseCov.channels == 1 &&
            measurementNoiseCov.rows() == measureParams &&
            measurementNoiseCov.cols() == measureParams)
        {
            S = mat_add(S, measurementNoiseCov);
        }

        Matf S_inv;
        if (!mat_inv(S, S_inv))
        {
            statePost = statePre;
            errorCovPost = errorCovPre;
            return statePost;
        }

        // K = P' H^T S^{-1}
        gain = mat_mul(PHt, S_inv); // (d x m)

        // x = x' + K y
        Matf Ky = mat_mul(gain, y); // (d x 1)
        statePost = mat_add(statePre, Ky);

        // P = (I - K H) P'
        Matf I = Matf::eye(dynamParams);
        Matf KH = mat_mul(gain, measurementMatrix); // (d x d)
        Matf I_KH = mat_sub(I, KH);
        errorCovPost = mat_mul(I_KH, errorCovPre);

        return statePost;
    }
}
