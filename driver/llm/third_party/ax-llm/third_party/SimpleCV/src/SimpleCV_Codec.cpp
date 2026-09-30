#include "SimpleCV.hpp"
#include "SimpleCV_Common.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#include "stb_image.h"
#include "stb_image_write.h"
#include <cctype>
#include <algorithm>
#include <filesystem>
#include <string_view>

namespace SimpleCV
{
    Mat imread(const std::string &filename, ColorSpace flag)
    {
        int w = 0, h = 0, c = 0;
        const int req_c = desired_channels(flag);

        unsigned char *p = stbi_load(filename.c_str(), &w, &h, &c, req_c);
        if (!p)
            return Mat();

        int out_c = (req_c != 0) ? req_c : c;

        Mat m;
        m.owner_ = std::shared_ptr<unsigned char>(p, [](unsigned char *ptr)
                                                  { stbi_image_free(ptr); });
        m.height = h;
        m.width = w;
        m.channels = out_c;
        m.step = w * out_c;
        m.data = m.owner_.get();

        // stb 输出 3/4 通道默认 RGB/RGBA，如果用户要 BGR/BGRA，则交换 R/B
        if (flag == ColorSpace::BGR || flag == ColorSpace::BGRA)
        {
            swap_rb_inplace(m);
        }

        return m;
    }

    Mat imdecode(const std::vector<unsigned char> &buf, ColorSpace flag)
    {
        if (buf.empty())
            return Mat();

        int w = 0, h = 0, c = 0;
        const int req_c = desired_channels(flag);

        unsigned char *p = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc *>(buf.data()),
            static_cast<int>(buf.size()),
            &w, &h, &c, req_c);

        if (!p)
            return Mat();

        int out_c = (req_c != 0) ? req_c : c;

        Mat m;
        m.owner_ = std::shared_ptr<unsigned char>(p, [](unsigned char *ptr)
                                                  { stbi_image_free(ptr); });
        m.height = h;
        m.width = w;
        m.channels = out_c;
        m.step = w * out_c;
        m.data = m.owner_.get();

        if (flag == ColorSpace::BGR || flag == ColorSpace::BGRA)
        {
            swap_rb_inplace(m);
        }

        return m;
    }
}

namespace SimpleCV
{
    static inline std::string to_lower(std::string s)
    {
        for (char &ch : s)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    }

    static inline std::string file_ext_lower(const std::string &filename)
    {
        auto pos = filename.find_last_of('.');
        if (pos == std::string::npos)
            return "";
        return to_lower(filename.substr(pos + 1));
    }

    // stb 写入内存回调
    static void stb_write_to_vector(void *context, void *data, int size)
    {
        auto *v = reinterpret_cast<std::vector<unsigned char> *>(context);
        auto *bytes = reinterpret_cast<unsigned char *>(data);
        v->insert(v->end(), bytes, bytes + size);
    }

    bool imencode(const Mat &mat, std::vector<unsigned char> &buf)
    {
        buf.clear();
        if (mat.empty())
            return false;
        if (mat.depth != Depth::U8 || mat.elem_size != 1)
            return false;

        // 默认用 PNG（无损、通用）
        // stride 是每行字节数
        int ok = stbi_write_png_to_func(
            stb_write_to_vector,
            &buf,
            mat.width,
            mat.height,
            mat.channels,
            mat.data,
            mat.step);
        return ok ? true : false;
    }

    bool imwrite(const std::string &filename, const Mat &mat)
    {
        if (mat.empty())
            return false;
        if (mat.depth != Depth::U8 || mat.elem_size != 1)
            return false;

        const std::string ext = file_ext_lower(filename);
        int ok = 0;

        if (ext == "png")
        {
            ok = stbi_write_png(filename.c_str(), mat.width, mat.height, mat.channels, mat.data, mat.step);
        }
        else if (ext == "jpg" || ext == "jpeg")
        {
            // jpg 质量默认 95
            // 注意：stb 写 jpg 不支持带 alpha；如果 channels==4 这里会失败/结果异常，通常要先转成 3 通道
            ok = stbi_write_jpg(filename.c_str(), mat.width, mat.height, mat.channels, mat.data, 95);
        }
        else if (ext == "bmp")
        {
            ok = stbi_write_bmp(filename.c_str(), mat.width, mat.height, mat.channels, mat.data);
        }
        else if (ext == "tga")
        {
            ok = stbi_write_tga(filename.c_str(), mat.width, mat.height, mat.channels, mat.data);
        }
        else
        {
            // 未知扩展名：默认写 png
            ok = stbi_write_png(filename.c_str(), mat.width, mat.height, mat.channels, mat.data, mat.step);
        }

        return ok ? true : false;
    }
}
