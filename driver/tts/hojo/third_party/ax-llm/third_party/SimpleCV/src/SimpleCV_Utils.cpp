#include "SimpleCV.hpp"
#include <filesystem>

namespace fs = std::filesystem;

namespace SimpleCV
{

    static inline bool is_sep(char c)
    {
#ifdef _WIN32
        return c == '/' || c == '\\';
#else
        return c == '/';
#endif
    }

    static inline std::string normalize_seps(std::string s)
    {
#ifdef _WIN32
        // filesystem 能处理两种分隔符，但我们内部统一成 '\\' 或 '/'
        // 这里统一成 fs::path::preferred_separator
        for (auto &ch : s)
        {
            if (ch == '/' || ch == '\\')
                ch = fs::path::preferred_separator;
        }
#else
        for (auto &ch : s)
        {
            if (ch == '\\')
                ch = '/';
        }
#endif
        return s;
    }

    // 简单通配符匹配：支持 '*' '?'，不支持字符组 []（你如果需要我也能加）
    static bool match_wildcard(std::string_view str, std::string_view pat)
    {
        size_t s = 0, p = 0;
        size_t star = std::string_view::npos, match = 0;

        while (s < str.size())
        {
            if (p < pat.size() && (pat[p] == '?' || pat[p] == str[s]))
            {
                ++s;
                ++p;
            }
            else if (p < pat.size() && pat[p] == '*')
            {
                star = p++;
                match = s;
            }
            else if (star != std::string_view::npos)
            {
                p = star + 1;
                s = ++match;
            }
            else
            {
                return false;
            }
        }
        while (p < pat.size() && pat[p] == '*')
            ++p;
        return p == pat.size();
    }

    static bool has_wildcards(const std::string &s)
    {
        return s.find('*') != std::string::npos || s.find('?') != std::string::npos;
    }

    // 把 pattern 拆成 path 部分（base dir）+ 余下的段（segments）
    // 例如 "a/b/**/c*.png" -> base="a/b", segments=["**","c*.png"]
    static void split_pattern(const std::string &pattern,
                              fs::path &base_dir,
                              std::vector<std::string> &segments)
    {
        std::string pat = normalize_seps(pattern);

        // 先按分隔符拆段
        std::vector<std::string> parts;
        std::string cur;
        for (char ch : pat)
        {
            if (is_sep(ch))
            {
                if (!cur.empty())
                {
                    parts.push_back(cur);
                    cur.clear();
                }
            }
            else
            {
                cur.push_back(ch);
            }
        }
        if (!cur.empty())
            parts.push_back(cur);

        // base_dir：直到遇到第一个含通配符的段（或 **）
        base_dir.clear();
#ifdef _WIN32
        // Windows: 处理盘符/UNC 前缀由 filesystem 自己解析更可靠
        // 如果 pattern 里有盘符，filesystem 会把 "C:" 当作一个段
#endif

        size_t i = 0;

        // 如果 pattern 是绝对路径，保留根
        fs::path full = fs::path(pat);
        if (full.is_absolute())
        {
            // 这里拿 root_path（比如 "C:\\" 或 "/"），再从相对部分开始拆
            base_dir = full.root_path();
            // 对绝对路径，full.relative_path() 会去掉 root
            // 我们用 full.relative_path() 重新拆段会更复杂，这里简化：
            // 直接用 parts 继续往下拼（盘符已在 root_path 里时，parts 里可能仍含 "C:"）
            // 保险起见：如果 base_dir 有 root_name/root_directory，跳过 parts 开头的 root_name 可能出现的重复
            // （这块不是很完美，但一般 pattern 传入 fs 能正确解析）
        }

        // base_dir 从头拼到第一个通配段之前
        for (; i < parts.size(); ++i)
        {
            const auto &seg = parts[i];
            if (seg == "**" || has_wildcards(seg))
                break;
            base_dir /= seg;
        }

        // 余下都作为 segments
        for (; i < parts.size(); ++i)
            segments.push_back(parts[i]);

        if (base_dir.empty())
            base_dir = ".";
    }

    static void glob_impl(const fs::path &dir,
                          const std::vector<std::string> &segs,
                          size_t idx,
                          bool recursive_double_star,
                          std::vector<std::string> &out)
    {
        if (idx >= segs.size())
        {
            // 到了末尾：dir 本身可能是文件/目录，返回实际存在的路径
            std::error_code ec;
            if (fs::exists(dir, ec))
                out.push_back(dir.string());
            return;
        }

        const std::string &seg = segs[idx];

        if (seg == "**")
        {
            if (!recursive_double_star)
            {
                // 当成普通目录名处理
                fs::path next = dir / seg;
                glob_impl(next, segs, idx + 1, recursive_double_star, out);
                return;
            }

            // "**" 两种展开：
            // 1) 匹配 0 层：直接跳过 "**"
            glob_impl(dir, segs, idx + 1, recursive_double_star, out);

            // 2) 匹配多层：递归遍历所有子目录，把它们当作 dir 继续匹配 "**"（idx 不变）
            std::error_code ec;
            if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
                return;

            for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
                 it != end; it.increment(ec))
            {
                if (ec)
                    break;
                if (it->is_directory(ec))
                {
                    glob_impl(it->path(), segs, idx + 1, recursive_double_star, out);
                }
            }
            return;
        }

        std::error_code ec;
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
            return;

        // 普通段：遍历当前目录的 direct children，看名字是否匹配
        for (fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end;
             it != end; it.increment(ec))
        {
            if (ec)
                break;
            const fs::path p = it->path();
            const std::string name = p.filename().string();

            if (match_wildcard(name, seg))
            {
                glob_impl(p, segs, idx + 1, recursive_double_star, out);
            }
        }
    }

    std::vector<std::string> glob(const std::string &pattern, bool recursive_double_star)
    {
        fs::path base;
        std::vector<std::string> segs;
        split_pattern(pattern, base, segs);

        std::vector<std::string> results;
        glob_impl(base, segs, 0, recursive_double_star, results);

        // 清理：只保留最终匹配到“实际存在的文件或目录”
        // （上面末尾已经 exists 了，这里主要做去重+排序）
        std::sort(results.begin(), results.end());
        results.erase(std::unique(results.begin(), results.end()), results.end());
        return results;
    }

} // namespace SimpleCV
