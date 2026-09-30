#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class Gemma3Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string image_pad_token = "<start_of_image>";

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
        text << "<bos>";
        
        // Extract system message if present
        std::string system_prefix;
        size_t start_idx = 0;
        
        if (!contents.empty() && contents[0].role == SYSTEM)
        {
            system_prefix = contents[0].data + "\n\n";
            start_idx = 1;
        }
        
        bool first_user = true;
        for (size_t i = start_idx; i < contents.size(); i++)
        {
            const auto &content = contents[i];
            
            if (content.role == USER)
            {
                switch (content.type)
                {
                case TEXT:
                    text << "<start_of_turn>user\n";
                    if (first_user)
                    {
                        text << system_prefix;
                        first_user = false;
                    }
                    text << content.data << "<end_of_turn>\n";
                    break;
                case IMAGE:
                    text << "<start_of_turn>user\n";
                    if (first_user)
                    {
                        text << system_prefix;
                        first_user = false;
                    }
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << image_pad_token;
                    }
                    text << content.data << "<end_of_turn>\n";
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
                    text << "<start_of_turn>model\n"
                         << cleaned_data << "<end_of_turn>\n";
                }
                else
                {
                    text << "<start_of_turn>model\n"
                         << content.data << "<end_of_turn>\n";
                }
            }
        }

        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "<start_of_turn>model\n";
        }

        return text.str();
    }
};

using gemma3_tokenizer = Gemma3Tokenizer<TEXT>;
REGISTER(Gemma3, gemma3_tokenizer)

using gemma3vl_tokenizer = Gemma3Tokenizer<TEXT, IMAGE>;
REGISTER(Gemma3VL, gemma3vl_tokenizer)
