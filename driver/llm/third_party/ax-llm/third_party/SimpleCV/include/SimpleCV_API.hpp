#ifndef SIMPLECV_API_HPP
#define SIMPLECV_API_HPP

#include "SimpleCV_Types.hpp"

namespace SimpleCV
{
    // ==========================================================
    // 线性卡尔曼滤波（OpenCV KalmanFilter 风格）
    // x(k) = A x(k-1) + B u(k)
    // z(k) = H x(k) + v(k)
    // ==========================================================
    class SIMPLECV_API KalmanFilter
    {
    public:
        int dynamParams = 0;   // state dimension
        int measureParams = 0; // measurement dimension
        int controlParams = 0; // control dimension

        Matf statePre;  // x'(k)
        Matf statePost; // x(k)

        Matf transitionMatrix;   // A
        Matf controlMatrix;      // B
        Matf measurementMatrix;  // H

        Matf processNoiseCov;     // Q
        Matf measurementNoiseCov; // R

        Matf errorCovPre;  // P'(k)
        Matf errorCovPost; // P(k)

        Matf gain; // K(k)

        KalmanFilter() = default;
        KalmanFilter(int dynam_params, int measure_params, int control_params = 0)
        {
            init(dynam_params, measure_params, control_params);
        }

        void init(int dynam_params, int measure_params, int control_params = 0);

        const Matf &predict(const Matf &control = Matf());
        const Matf &correct(const Matf &measurement);
    };

    // imgcodec
    SIMPLECV_API Mat imread(const std::string &filename, ColorSpace flag = ColorSpace::UNCHANGED);
    SIMPLECV_API Mat imdecode(const std::vector<unsigned char> &buf, ColorSpace flag = ColorSpace::UNCHANGED);

    SIMPLECV_API bool imwrite(const std::string &filename, const Mat &mat);
    SIMPLECV_API bool imencode(const Mat &mat, std::vector<unsigned char> &buf);

    // imgproc
    SIMPLECV_API void resize(const Mat &src, Mat &dst, int dst_width, int dst_height);

    SIMPLECV_API void cvtColor(const Mat &src, Mat &dst, ColorSpace dst_space, ColorSpace src_space = ColorSpace::AUTO);
    SIMPLECV_API Mat cvtColor(const Mat &src, ColorSpace dst_space, ColorSpace src_space = ColorSpace::AUTO);

    SIMPLECV_API void copyMakeBorder(
        const Mat &src,
        Mat &dst,
        int top, int bottom, int left, int right,
        BorderType borderType = BorderType::CONSTANT,
        const std::vector<unsigned char> &value = std::vector<unsigned char>{0, 0, 0, 255});

    SIMPLECV_API void rectangle(Mat &img, Point pt1, Point pt2, const Scalar &color,
                                int thickness = 1, int lineType = 8, int shift = 0);
    SIMPLECV_API void rectangle(Mat &img, Rect rec, const Scalar &color,
                                int thickness = 1);

    SIMPLECV_API void circle(Mat &img, Point center, int radius, const Scalar &color,
                             int thickness = 1);

    SIMPLECV_API void line(Mat &img, Point p0, Point p1, const Scalar &color,
                           int thickness = 1);

    SIMPLECV_API void putText(Mat &img,
                              const std::string &text,
                              Point org,
                              double fontScale,
                              const Scalar &color,
                              int thickness = 1,
                              int lineType = 8,
                              bool bottomLeftOrigin = false);

    SIMPLECV_API Size getTextSize(const std::string &text,
                                  double fontScale,
                                  int thickness,
                                  int *baseLine);

    SIMPLECV_API std::vector<std::string> glob(const std::string &pattern, bool recursive_double_star = true);

    // ==========================================================
    // 仿射变换（人脸对齐 / CNN 前处理）
    // M 为 2x3 forward affine：dst_pt = M * [src_pt; 1]
    // 内部对 M 求逆做 dst→src 反向采样 + 双线性插值
    // ==========================================================
    SIMPLECV_API void warpAffine(
        const Mat &src,
        Mat &dst,
        const Matf &M,
        Size dsize,
        BorderType borderType = BorderType::CONSTANT,
        const Scalar &borderValue = Scalar());

    // 由 N(>=2) 对关键点估计 4-DOF 相似变换（旋转+均匀缩放+平移）
    // 返回 2x3 Matf；当点数不足或退化时返回 empty Matf
    // 典型用法：人脸 5 点对齐到 arcface 模板
    SIMPLECV_API Matf estimateAffinePartial2D(
        const std::vector<Point2f> &from,
        const std::vector<Point2f> &to);
}

#endif // SIMPLECV_API_HPP

