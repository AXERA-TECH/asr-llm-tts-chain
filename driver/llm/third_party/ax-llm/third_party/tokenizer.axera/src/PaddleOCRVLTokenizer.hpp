#pragma once
#include <sstream>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

// PaddleOCR-VL-1.5 chat template:
//   <|begin_of_sentence|>{system}\nUser: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>...<|IMAGE_END|>{text}\nAssistant:\n{response}</s>
template <ContentType... Types>
class PaddleOCRVLTokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string image_pad_token = "<|IMAGE_PLACEHOLDER|>";
    std::string img_start_token = "<|IMAGE_START|>";
    std::string img_end_token = "<|IMAGE_END|>";
    std::string bos_token = "<|begin_of_sentence|>";
    std::string eos_token = "</s>";

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
                text << content.data << "\n";
            }
            else if (content.role == USER)
            {
                text << "User: ";
                switch (content.type)
                {
                case IMAGE:
                    text << img_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << image_pad_token;
                    }
                    text << img_end_token;
                    text << content.data << "\n";
                    break;
                case VIDEO:
                    text << img_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << image_pad_token;
                    }
                    text << img_end_token;
                    text << content.data << "\n";
                    break;
                case TEXT:
                default:
                    text << content.data << "\n";
                    break;
                }
            }
            else if (content.role == ASSISTANT)
            {
                text << "Assistant:\n"
                     << content.data << eos_token;
            }
        }

        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "Assistant:\n";
        }

        return text.str();
    }
};

using paddleocrvl_tokenizer = PaddleOCRVLTokenizer<TEXT, IMAGE, VIDEO>;
REGISTER(PaddleOCRVL, paddleocrvl_tokenizer)
