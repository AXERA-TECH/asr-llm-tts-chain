#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class MiniCPM4Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string image_pad_token = "<image>";
    std::string audio_start_token = "<|audio_start|>";
    std::string audio_end_token = "<|audio_end|>";
    std::string audio_pad_token = "<|audio|>";

public:
    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        // check contents type
        for (const auto &content : contents)
        {
            if (!this->support(content.type))
            {
                ALOGE("unsupport content type: %d", content.type);
                return {};
            }
        }

        std::stringstream text;
        
        // Add BOS token at the beginning
        text << "<s>";
        for (const auto &content : contents)
        {
            if (content.role == SYSTEM)
            {
                text << "<|im_start|>system\n"
                     << content.data << "<|im_end|>\n";
            }
            else if (content.role == USER)
            {
                switch (content.type)
                {
                case TEXT:
                    text << "<|im_start|>user\n"
                         << content.data << "<|im_end|>\n";
                    break;
                case IMAGE:
                    text << "<|im_start|>user\n"
                         << image_pad_token << "\n"
                         << content.data << "<|im_end|>\n";
                    break;
                case AUDIO:
                    text << "<|im_start|>user\n"
                         << audio_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << audio_pad_token;
                    }
                    text << audio_end_token << "\n"
                         << content.data << "<|im_end|>\n";
                    break;
                default:
                    break;
                }
            }
            else if (content.role == ASSISTANT)
            {
                if (!this->think_in_prompt)
                {
                    auto cleaned_data = this->remove_thinking(content.data);
                    text << "<|im_start|>assistant\n"
                         << cleaned_data << "<|im_end|>\n";
                }
                else
                {
                    text << "<|im_start|>assistant\n"
                         << content.data << "<|im_end|>\n";
                }
            }
        }

        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n";
        }

        return text.str();
    }
};

using minicpm4_tokenizer = MiniCPM4Tokenizer<TEXT>;
REGISTER(MiniCPM4, minicpm4_tokenizer)

using minicpmv4_tokenizer = MiniCPM4Tokenizer<TEXT, IMAGE>;
REGISTER(MiniCPMV4, minicpmv4_tokenizer)

using minicpmo4_5_tokenizer = MiniCPM4Tokenizer<TEXT, IMAGE, AUDIO>;
REGISTER(MiniCPMO4_5, minicpmo4_5_tokenizer)
