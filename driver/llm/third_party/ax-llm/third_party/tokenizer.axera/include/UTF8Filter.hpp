#pragma once
#include <string>

class UTF8Filter
{
private:
    std::string utf8_buffer = "";

    static bool is_cont(unsigned char c)
    {
        return (c & 0xC0) == 0x80;
    }

    bool next_valid_span(size_t pos, size_t &span_len, bool &need_more) const
    {
        need_more = false;
        span_len = 0;
        if (pos >= utf8_buffer.size())
            return false;

        const unsigned char c0 = static_cast<unsigned char>(utf8_buffer[pos]);
        if (c0 < 0x80)
        {
            span_len = 1;
            return true;
        }

        if ((c0 & 0xE0) == 0xC0)
        {
            if (pos + 1 >= utf8_buffer.size()) { need_more = true; return false; }
            const unsigned char c1 = static_cast<unsigned char>(utf8_buffer[pos + 1]);
            if (is_cont(c1) && c0 >= 0xC2)
            {
                span_len = 2;
                return true;
            }
            return false;
        }

        if ((c0 & 0xF0) == 0xE0)
        {
            if (pos + 2 >= utf8_buffer.size()) { need_more = true; return false; }
            const unsigned char c1 = static_cast<unsigned char>(utf8_buffer[pos + 1]);
            const unsigned char c2 = static_cast<unsigned char>(utf8_buffer[pos + 2]);
            if (!is_cont(c1) || !is_cont(c2))
                return false;
            if (c0 == 0xE0 && c1 < 0xA0)
                return false;
            if (c0 == 0xED && c1 >= 0xA0)
                return false;
            span_len = 3;
            return true;
        }

        if ((c0 & 0xF8) == 0xF0)
        {
            if (pos + 3 >= utf8_buffer.size()) { need_more = true; return false; }
            const unsigned char c1 = static_cast<unsigned char>(utf8_buffer[pos + 1]);
            const unsigned char c2 = static_cast<unsigned char>(utf8_buffer[pos + 2]);
            const unsigned char c3 = static_cast<unsigned char>(utf8_buffer[pos + 3]);
            if (!is_cont(c1) || !is_cont(c2) || !is_cont(c3))
                return false;
            if (c0 == 0xF0 && c1 < 0x90)
                return false;
            if (c0 == 0xF4 && c1 > 0x8F)
                return false;
            if (c0 < 0xF0 || c0 > 0xF4)
                return false;
            span_len = 4;
            return true;
        }

        return false;
    }

    std::string drain_valid_utf8(bool drop_incomplete_tail)
    {
        std::string out;
        size_t pos = 0;
        while (pos < utf8_buffer.size())
        {
            size_t span_len = 0;
            bool need_more = false;
            if (next_valid_span(pos, span_len, need_more))
            {
                out.append(utf8_buffer, pos, span_len);
                pos += span_len;
                continue;
            }

            if (need_more && !drop_incomplete_tail)
                break;

            ++pos;
        }

        utf8_buffer.erase(0, pos);
        if (drop_incomplete_tail)
            utf8_buffer.clear();
        return out;
    }

public:
    std::string filter(const std::string &str)
    {
        utf8_buffer += str;
        return drain_valid_utf8(false);
    }

    std::string flush()
    {
        return drain_valid_utf8(true);
    }
};
