#pragma once

#include <sstream>
#include <string>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class Gemma4Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string bos_token = "<bos>";
    std::string turn_token = "<|turn>";
    std::string turn_end_token = "<turn|>";
    std::string boi_token = "<|image>";
    std::string eoi_token = "<image|>";
    std::string image_pad_token = "<|image|>";
    std::string boa_token = "<|audio>";
    std::string eoa_token = "<audio|>";
    std::string video_pad_token = "<|video|>";
    std::string audio_pad_token = "<|audio|>";
    std::string think_token = "<|think|>";

    static std::string strip_channel_sections(const std::string &text)
    {
        std::string out;
        out.reserve(text.size());

        size_t pos = 0;
        while (pos < text.size())
        {
            size_t think_pos = text.find("<|channel", pos);
            size_t think_tag_pos = text.find("<|think|>", pos);
            size_t start = std::string::npos;
            size_t skip_len = 0;

            if (think_pos != std::string::npos)
            {
                start = think_pos;
            }
            if (think_tag_pos != std::string::npos && (start == std::string::npos || think_tag_pos < start))
            {
                start = think_tag_pos;
            }

            if (start == std::string::npos)
            {
                out.append(text, pos, std::string::npos);
                break;
            }

            out.append(text, pos, start - pos);
            skip_len = (start == think_tag_pos) ? std::string("<|think|>").size() : 0;

            size_t end = text.find("<channel|>", start + skip_len);
            if (end == std::string::npos)
            {
                break;
            }
            pos = end + std::string("<channel|>").size();
        }

        BaseMixinTokenizer<Types...>::trim_inplace(out);
        return out;
    }

    void append_turn_open(std::stringstream &text, const std::string &role) const
    {
        text << turn_token << role << "\n";
    }

    void append_turn_close(std::stringstream &text) const
    {
        text << turn_end_token << "\n";
    }

    template <typename EmitTokenFn>
    void append_media_placeholders(std::stringstream &text,
                                   int num_media,
                                   int num_media_tokens,
                                   const std::string &prefix_token,
                                   const std::string &suffix_token,
                                   EmitTokenFn emit_token) const
    {
        if (num_media <= 0 || num_media_tokens <= 0)
            return;

        text << "\n\n";
        for (int media_idx = 0; media_idx < num_media; ++media_idx)
        {
            text << prefix_token;
            for (int token_idx = 0; token_idx < num_media_tokens; ++token_idx)
            {
                emit_token();
            }
            text << suffix_token;
            if (media_idx + 1 < num_media)
                text << "\n\n";
        }
    }

public:
    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        for (const auto &content : contents)
        {
            if (!this->support(content.type))
            {
                ALOGE("unsupport content type: %d", content.type);
                return {};
            }
        }

        std::stringstream text;
        text << bos_token;

        for (const auto &content : contents)
        {
            if (content.role == SYSTEM)
            {
                append_turn_open(text, "system");
                text << content.data;
                append_turn_close(text);
                continue;
            }

            if (content.role == USER)
            {
                append_turn_open(text, "user");
                switch (content.type)
                {
                case TEXT:
                    text << content.data;
                    break;
                case IMAGE:
                    append_media_placeholders(text,
                                              content.num_media,
                                              content.num_media_tokens,
                                              boi_token,
                                              eoi_token,
                                              [&text, this]()
                                              {
                                                  text << image_pad_token;
                                              });
                    if (!content.data.empty())
                    {
                        if (content.num_media > 0) text << "\n";
                        text << content.data;
                    }
                    break;
                case VIDEO:
                    append_media_placeholders(text,
                                              content.num_media,
                                              content.num_media_tokens,
                                              boi_token,
                                              eoi_token,
                                              [&text, this]()
                                              {
                                                  text << video_pad_token;
                                              });
                    if (!content.data.empty())
                    {
                        if (content.num_media > 0) text << "\n";
                        text << content.data;
                    }
                    break;
                case AUDIO:
                    append_media_placeholders(text,
                                              content.num_media,
                                              content.num_media_tokens,
                                              boa_token,
                                              eoa_token,
                                              [&text, this]()
                                              {
                                                  text << audio_pad_token;
                                              });
                    if (!content.data.empty())
                    {
                        if (content.num_media > 0) text << "\n";
                        text << content.data;
                    }
                    break;
                default:
                    break;
                }
                append_turn_close(text);
                continue;
            }

            if (content.role == ASSISTANT)
            {
                append_turn_open(text, "model");
                if (this->think_in_prompt)
                {
                    text << content.data;
                }
                else
                {
                    text << strip_channel_sections(content.data);
                }
                append_turn_close(text);
            }
        }

        if (!contents.empty() && contents.back().role == USER && add_generation_prompt)
        {
            append_turn_open(text, "model");
        }

        return text.str();
    }
};

using gemma4_tokenizer = Gemma4Tokenizer<TEXT>;
REGISTER(Gemma4, gemma4_tokenizer)

using gemma4vl_tokenizer = Gemma4Tokenizer<TEXT, IMAGE, VIDEO, AUDIO>;
REGISTER(Gemma4VL, gemma4vl_tokenizer)
