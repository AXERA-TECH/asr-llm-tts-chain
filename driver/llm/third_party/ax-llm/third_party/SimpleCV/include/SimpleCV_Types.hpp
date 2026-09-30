#ifndef SIMPLECV_TYPES_HPP
#define SIMPLECV_TYPES_HPP

// ==========================================================
// 跨平台：Windows/Linux 导出宏 + 常见宏污染处理
// ==========================================================
#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
// 防止 Windows 头文件把 std::min/std::max 变成宏
#define NOMINMAX
#endif

#if defined(SIMPLECV_BUILD_DLL)
#define SIMPLECV_API __declspec(dllexport)
#elif defined(SIMPLECV_USE_DLL)
#define SIMPLECV_API __declspec(dllimport)
#else
#define SIMPLECV_API
#endif
#else
// GCC/Clang: 可见性控制（不强制，留空也可以）
#if defined(__GNUC__) && __GNUC__ >= 4
#define SIMPLECV_API __attribute__((visibility("default")))
#else
#define SIMPLECV_API
#endif
#endif

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility> // std::move
#include <climits>

namespace SimpleCV
{
    // 像素格式/通道排列
    enum class ColorSpace
    {
        AUTO = 0,  // cvtColor 用：自动根据 channels 推断（1->GRAY,3->RGB,4->RGBA）
        UNCHANGED, // imread 用：不改 channels，保持 stb 解码结果（通常是 1/2/3/4）
        GRAY,      // 1 channel
        RGB,       // 3 channel
        BGR,       // 3 channel
        RGBA,      // 4 channel
        BGRA       // 4 channel
    };

    enum class BorderType
    {
        CONSTANT,   // 常量填充
        REPLICATE,  // 边缘复制 (aaaa|abcd|dddd)
        REFLECT,    // 镜像 (dcba|abcd|dcba)
        REFLECT_101 // 镜像101 (cbab|abcd|cbab) 也叫 reflect without repeating edge
    };

    // 元素类型（类似 OpenCV depth；这里只做最小集合，后续可扩展）
    enum class Depth
    {
        U8 = 0,
        I32,
        F32,
        F64,
    };

    static inline int depth_bytes(Depth d)
    {
        switch (d)
        {
        case Depth::U8:
            return 1;
        case Depth::I32:
            return 4;
        case Depth::F32:
            return 4;
        case Depth::F64:
            return 8;
        default:
            return 0;
        }
    }

    static inline int make_type(Depth d, int channels)
    {
        // Layout: low 8 bits = depth, high 8 bits = channels
        return (static_cast<int>(d) & 0xFF) | ((channels & 0xFF) << 8);
    }

    static inline Depth type_depth(int type)
    {
        return static_cast<Depth>(type & 0xFF);
    }

    static inline int type_channels(int type)
    {
        return (type >> 8) & 0xFF;
    }

    template <typename T>
    struct DepthOf
    {
    private:
        using U = typename std::remove_cv<T>::type;

        static constexpr bool kIsU8 =
            std::is_same<U, std::uint8_t>::value ||
            std::is_same<U, unsigned char>::value;

        static constexpr bool kIsI32 =
            std::is_same<U, std::int32_t>::value ||
            (std::is_same<U, int>::value && sizeof(int) == 4);

        static constexpr bool kIsF32 = std::is_same<U, float>::value;
        static constexpr bool kIsF64 = std::is_same<U, double>::value;

    public:
        static constexpr bool supported = kIsU8 || kIsI32 || kIsF32 || kIsF64;
        static_assert(supported, "SimpleCV::Mat_ only supports u8/i32/f32/f64 element types.");

        static constexpr Depth value =
            kIsU8 ? Depth::U8 :
            kIsI32 ? Depth::I32 :
            kIsF32 ? Depth::F32 :
            Depth::F64;
    };

    template <typename _Tp>
    static inline _Tp saturate_cast(int v)
    {
        return _Tp(v);
    }
    template <>
    inline unsigned char saturate_cast<unsigned char>(int v)
    {
        return (unsigned char)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX
                                                                    : 0);
    }

    template <typename _Tp>
    struct Scalar_
    {
        Scalar_()
        {
            v[0] = 0;
            v[1] = 0;
            v[2] = 0;
            v[3] = 0;
        }
        Scalar_(_Tp _v0)
        {
            v[0] = _v0;
            v[1] = 0;
            v[2] = 0;
            v[3] = 0;
        }
        Scalar_(_Tp _v0, _Tp _v1)
        {
            v[0] = _v0;
            v[1] = _v1;
            v[2] = 0;
            v[3] = 0;
            width = v[0];
            height = v[1];
        }
        Scalar_(_Tp _v0, _Tp _v1, _Tp _v2)
        {
            v[0] = _v0;
            v[1] = _v1;
            v[2] = _v2;
            v[3] = 0;
            width = v[0];
            height = v[1];
        }
        Scalar_(_Tp _v0, _Tp _v1, _Tp _v2, _Tp _v3)
        {
            v[0] = _v0;
            v[1] = _v1;
            v[2] = _v2;
            v[3] = _v3;
            width = v[0];
            height = v[1];
        }

        const _Tp operator[](const int i) const { return v[i]; }
        _Tp &operator[](const int i) { return v[i]; }

        _Tp v[4];

        // NOTE: 不要用 v[] 做默认成员初始化（v 未初始化会导致 UB）
        int width = 0;
        int height = 0;
    };

    typedef Scalar_<unsigned char> Scalar;

    template <typename _Tp>
    struct Point_
    {
        Point_() : x(0), y(0) {}
        Point_(_Tp _x, _Tp _y) : x(_x), y(_y) {}

        template <typename _Tp2>
        operator Point_<_Tp2>() const
        {
            return Point_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y));
        }

        bool operator==(const Point_ &b) { return x == b.x && y == b.y; }
        bool operator!=(const Point_ &b) { return x != b.x || y != b.y; }
        Point_<_Tp> operator-(const Point_<_Tp> &b)
        {
            return Point_<_Tp>(x - b.x, y - b.y);
        }

        _Tp x;
        _Tp y;
    };

    typedef Point_<int> Point;
    typedef Point_<float> Point2f;

    template <typename _Tp>
    struct Size_
    {
        Size_() : width(0), height(0) {}
        Size_(_Tp _w, _Tp _h) : width(_w), height(_h) {}
        Size_(_Tp _w, _Tp _h, _Tp _c) : width(_w), height(_h), channel(_c) {}

        template <typename _Tp2>
        operator Size_<_Tp2>() const
        {
            return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
        }

        _Tp width;
        _Tp height;
        _Tp channel;
    };

    typedef Size_<int> Size;
    typedef Size_<float> Size2f;

    template <typename _Tp>
    struct Rect_
    {
        Rect_() : x(0), y(0), width(0), height(0) {}
        Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h) : x(_x), y(_y), width(_w), height(_h) {}
        Rect_(Point_<_Tp> _p, Size_<_Tp> _size)
            : x(_p.x), y(_p.y), width(_size.width), height(_size.height) {}

        template <typename _Tp2>
        operator Rect_<_Tp2>() const
        {
            return Rect_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y),
                               saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
        }

        _Tp x;
        _Tp y;
        _Tp width;
        _Tp height;

        _Tp area() const { return width * height; }
    };

    template <typename _Tp>
    static inline Rect_<_Tp> &operator&=(Rect_<_Tp> &a, const Rect_<_Tp> &b)
    {
        _Tp x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
        a.width = std::min(a.x + a.width, b.x + b.width) - x1;
        a.height = std::min(a.y + a.height, b.y + b.height) - y1;
        a.x = x1;
        a.y = y1;
        if (a.width <= 0 || a.height <= 0)
            a = Rect_<_Tp>();
        return a;
    }

    template <typename _Tp>
    static inline Rect_<_Tp> &operator|=(Rect_<_Tp> &a, const Rect_<_Tp> &b)
    {
        _Tp x1 = std::min(a.x, b.x), y1 = std::min(a.y, b.y);
        a.width = std::max(a.x + a.width, b.x + b.width) - x1;
        a.height = std::max(a.y + a.height, b.y + b.height) - y1;
        a.x = x1;
        a.y = y1;
        return a;
    }

    template <typename _Tp>
    static inline Rect_<_Tp> operator&(const Rect_<_Tp> &a, const Rect_<_Tp> &b)
    {
        Rect_<_Tp> c = a;
        return c &= b;
    }

    template <typename _Tp>
    static inline Rect_<_Tp> operator|(const Rect_<_Tp> &a, const Rect_<_Tp> &b)
    {
        Rect_<_Tp> c = a;
        return c |= b;
    }

    typedef Rect_<int> Rect;
    typedef Rect_<float> Rect2f;

    class SIMPLECV_API Mat
    {
    public:
        int height = 0;
        int width = 0;
        int channels = 0;
        Depth depth = Depth::U8;
        int elem_size = 1; // bytes per channel element
        unsigned char *data = nullptr;
        int step = 0; // stride in bytes

        int rows() const { return height; }
        int cols() const { return width; }

        int elemSize1() const { return elem_size; }
        int elemSize() const { return channels * elem_size; }

        std::size_t total() const
        {
            if (empty())
                return 0;
            return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        }

        std::size_t totalBytes() const
        {
            if (empty())
                return 0;
            return static_cast<std::size_t>(height) * static_cast<std::size_t>(step);
        }

        bool isContinuous() const
        {
            if (empty())
                return false;
            const int min_step = width * channels * elem_size;
            return height <= 1 || step == min_step;
        }

        int type() const { return make_type(depth, channels); }

        Mat(int h, int w, int c, unsigned char *d, int s, bool is_own_data = false)
        {
            reset(h, w, c, d, s, is_own_data, Depth::U8);
        }

        Mat(int h, int w, int c, unsigned char *d, std::size_t s, bool is_own_data = false)
        {
            int si = (s > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                         ? std::numeric_limits<int>::max()
                         : static_cast<int>(s);
            reset(h, w, c, d, si, is_own_data, Depth::U8);
        }

        Mat(int h, int w, int c, unsigned char *d)
        {
            reset(h, w, c, d, w * c, false, Depth::U8);
        }

        Mat(int h, int w, int c, unsigned char *d, bool is_own_data)
        {
            reset(h, w, c, d, w * c, is_own_data, Depth::U8);
        }

        Mat(int h, int w, int c, int s)
        {
            create(h, w, c, Depth::U8, s);
        }

        Mat(int h, int w, int c)
        {
            create(h, w, c, Depth::U8, 0);
        }

        Mat() = default;

        Mat(const Mat &) = default;
        Mat &operator=(const Mat &) = default;

        Mat(Mat &&) noexcept = default;
        Mat &operator=(Mat &&) noexcept = default;

        ~Mat() = default;

        bool empty() const { return data == nullptr || height <= 0 || width <= 0 || channels <= 0; }

        unsigned char *ptr(int r = 0)
        {
            if (empty())
                return nullptr;
            return data + static_cast<std::size_t>(r) * static_cast<std::size_t>(step);
        }
        const unsigned char *ptr(int r = 0) const
        {
            if (empty())
                return nullptr;
            return data + static_cast<std::size_t>(r) * static_cast<std::size_t>(step);
        }

        template <typename _Tp>
        bool isType() const
        {
            return depth == DepthOf<_Tp>::value && elem_size == static_cast<int>(sizeof(_Tp));
        }

        template <typename _Tp>
        _Tp *ptr(int r = 0)
        {
            if (!isType<_Tp>() || empty())
                return nullptr;
            return reinterpret_cast<_Tp *>(data + static_cast<std::size_t>(r) * static_cast<std::size_t>(step));
        }
        template <typename _Tp>
        const _Tp *ptr(int r = 0) const
        {
            if (!isType<_Tp>() || empty())
                return nullptr;
            return reinterpret_cast<const _Tp *>(data + static_cast<std::size_t>(r) * static_cast<std::size_t>(step));
        }

        template <typename _Tp>
        _Tp *at_ptr(int r, int c, int k = 0)
        {
            _Tp *row = ptr<_Tp>(r);
            if (!row)
                return nullptr;
            return row + static_cast<std::size_t>(c) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(k);
        }
        template <typename _Tp>
        const _Tp *at_ptr(int r, int c, int k = 0) const
        {
            const _Tp *row = ptr<_Tp>(r);
            if (!row)
                return nullptr;
            return row + static_cast<std::size_t>(c) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(k);
        }

        Mat clone() const
        {
            if (empty())
                return Mat();
            Mat out;
            out.create(height, width, channels, depth, 0);
            const size_t row_bytes = static_cast<size_t>(width) * static_cast<size_t>(channels) * static_cast<size_t>(elem_size);
            for (int y = 0; y < height; ++y)
            {
                std::memcpy(out.data + y * out.step, data + y * step, row_bytes);
            }
            return out;
        }

        void create(int h, int w, int c, Depth d, int s = 0)
        {
            if (h <= 0 || w <= 0 || c <= 0)
            {
                release();
                return;
            }
            depth = d;
            elem_size = depth_bytes(depth);
            if (elem_size <= 0)
            {
                release();
                return;
            }

            const int min_step = w * c * elem_size;
            if (s <= 0)
                s = min_step;
            else if (s < min_step)
                s = min_step;

            const size_t bytes = static_cast<size_t>(h) * static_cast<size_t>(s);
            unsigned char *p = new unsigned char[bytes];

            std::shared_ptr<unsigned char> sp(p, [](unsigned char *ptr)
                                              { delete[] ptr; });

            owner_ = std::move(sp);
            height = h;
            width = w;
            channels = c;
            step = s;
            data = owner_.get();
        }

        void create(int h, int w, int c, int s)
        {
            create(h, w, c, Depth::U8, s);
        }

        void create(int h, int w, int c)
        {
            create(h, w, c, Depth::U8, 0);
        }

        void release()
        {
            owner_.reset();
            height = width = channels = step = 0;
            depth = Depth::U8;
            elem_size = 1;
            data = nullptr;
        }

    private:
        std::shared_ptr<unsigned char> owner_;

        void reset(int h, int w, int c, unsigned char *d, int s, bool is_own_data, Depth dpth)
        {
            if (h <= 0 || w <= 0 || c <= 0 || d == nullptr)
            {
                release();
                return;
            }

            depth = dpth;
            elem_size = depth_bytes(depth);
            if (elem_size <= 0)
            {
                release();
                return;
            }

            const int min_step = w * c * elem_size;
            if (s < min_step)
                s = min_step;

            if (is_own_data)
            {
                owner_ = std::shared_ptr<unsigned char>(d, [](unsigned char *ptr)
                                                        { delete[] ptr; });
            }
            else
            {
                owner_ = std::shared_ptr<unsigned char>(d, [](unsigned char *) {});
            }

            height = h;
            width = w;
            channels = c;
            step = s;
            data = d;
        }

        friend Mat imread(const std::string &filename, ColorSpace flag);
        friend Mat imdecode(const std::vector<unsigned char> &buf, ColorSpace flag);
    };

    template <typename _Tp>
    class Mat_ : public Mat
    {
    public:
        using value_type = _Tp;

        Mat_() = default;

        Mat_(int r, int c, int ch = 1)
        {
            create(r, c, ch);
        }

        Mat_(int r, int c, int ch, int step_bytes)
        {
            create(r, c, ch, step_bytes);
        }

        // 从 Mat 视图构造（不会做拷贝）
        explicit Mat_(const Mat &m) : Mat(m)
        {
            if (m.depth != DepthOf<_Tp>::value || m.elem_size != static_cast<int>(sizeof(_Tp)))
            {
                release();
            }
        }

        int rows() const { return height; }
        int cols() const { return width; }

        void create(int r, int c, int ch = 1)
        {
            Mat::create(r, c, ch, DepthOf<_Tp>::value, 0);
        }

        void create(int r, int c, int ch, int step_bytes)
        {
            Mat::create(r, c, ch, DepthOf<_Tp>::value, step_bytes);
        }

        _Tp *ptr(int r = 0)
        {
            return reinterpret_cast<_Tp *>(data + (size_t)r * (size_t)step);
        }
        const _Tp *ptr(int r = 0) const
        {
            return reinterpret_cast<const _Tp *>(data + (size_t)r * (size_t)step);
        }

        _Tp &at(int r, int c, int k = 0)
        {
            return ptr(r)[(size_t)c * (size_t)channels + (size_t)k];
        }
        _Tp at(int r, int c, int k = 0) const
        {
            return ptr(r)[(size_t)c * (size_t)channels + (size_t)k];
        }

        void setZero();
        void setIdentity(_Tp diag = (_Tp)1);

        static Mat_ eye(int n, _Tp diag = (_Tp)1);

        Mat_ clone() const { return Mat_(Mat::clone()); }
    };

    template <typename _Tp>
    inline void Mat_<_Tp>::setZero()
    {
        if (empty())
            return;
        const size_t bytes = static_cast<size_t>(height) * static_cast<size_t>(step);
        std::memset(data, 0, bytes);
    }

    template <typename _Tp>
    inline void Mat_<_Tp>::setIdentity(_Tp diag)
    {
        if (empty())
            return;
        if (channels != 1)
            return;
        setZero();
        int n = std::min(height, width);
        for (int i = 0; i < n; ++i)
            at(i, i) = diag;
    }

    template <typename _Tp>
    inline Mat_<_Tp> Mat_<_Tp>::eye(int n, _Tp diag)
    {
        Mat_<_Tp> m(n, n, 1);
        m.setIdentity(diag);
        return m;
    }

    // 常用 typedef（类似 OpenCV 的 Mat_ / Mat1f 风格）
    using Mat8u = Mat_<std::uint8_t>;
    using Mat32i = Mat_<std::int32_t>;
    using Mat32f = Mat_<float>;
    using Mat64f = Mat_<double>;

    // 兼容/简写
    using Matf = Mat32f;
}

#endif // SIMPLECV_TYPES_HPP
