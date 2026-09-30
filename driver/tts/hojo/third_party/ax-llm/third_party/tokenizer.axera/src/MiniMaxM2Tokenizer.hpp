#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class MiniMaxM2Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string image_pad_token = "]<]image[>[";
    std::string video_pad_token = "]<]video[>[";
    std::string speech_pad_token = "]<]speech[>[";

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
        
        // Find the last user message index for thinking logic
        int last_user_index = -1;
        for (size_t i = 0; i < contents.size(); i++)
        {
            if (contents[i].role == USER)
            {
                last_user_index = static_cast<int>(i);
            }
        }

        // Render system message
        text << "]~!b[]~b]system\n";
        bool has_system = false;
        size_t start_idx = 0;
        
        if (!contents.empty() && contents[0].role == SYSTEM)
        {
            text << contents[0].data;
            has_system = true;
            start_idx = 1;
        }
        
        if (!has_system)
        {
            text << "You are a helpful assistant. Your name is MiniMax-M2 and is built by MiniMax.";
        }
        
        text << "[e~[\n";

        // Process remaining messages
        for (size_t i = start_idx; i < contents.size(); i++)
        {
            const auto &content = contents[i];
            
            if (content.role == USER)
            {
                text << "]~b]user\n";
                
                switch (content.type)
                {
                case TEXT:
                    text << content.data;
                    break;
                case IMAGE:
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << image_pad_token;
                    }
                    text << content.data;
                    break;
                case VIDEO:
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << video_pad_token;
                    }
                    text << content.data;
                    break;
                case AUDIO:
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << speech_pad_token;
                    }
                    text << content.data;
                    break;
                default:
                    text << content.data;
                    break;
                }
                
                text << "[e~[\n";
            }
            else if (content.role == ASSISTANT)
            {
                text << "]~b]ai\n";
                
                // Handle thinking content
                std::string reasoning_content;
                std::string actual_content = content.data;
                
                // Extract reasoning content from <think> tags if present
                size_t think_start = content.data.find("<think>");
                size_t think_end = content.data.find("</think>");
                
                if (think_start != std::string::npos && think_end != std::string::npos && think_end > think_start)
                {
                    reasoning_content = content.data.substr(think_start + 7, think_end - think_start - 7);
                    actual_content = content.data.substr(think_end + 8);
                    size_t content_start = actual_content.find_first_not_of(" \n\r\t");
                    if (content_start != std::string::npos)
                    {
                        actual_content = actual_content.substr(content_start);
                    }
                }
                
                // Show thinking for messages after last user
                if (static_cast<int>(i) > last_user_index && !reasoning_content.empty())
                {
                    text << "<think>\n" << reasoning_content << "\n</think>\n\n";
                }
                
                if (!this->think_in_prompt)
                {
                    auto cleaned_data = this->remove_thinking(actual_content);
                    text << cleaned_data;
                }
                else
                {
                    text << actual_content;
                }
                
                text << "[e~[\n";
            }
        }

        // Add generation prompt
        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "]~b]ai\n<think>\n";
        }

        return text.str();
    }
};

// Register tokenizer variants
using minimaxm2_tokenizer = MiniMaxM2Tokenizer<TEXT>;
REGISTER(MiniMaxM2, minimaxm2_tokenizer)

using minimaxm2vl_tokenizer = MiniMaxM2Tokenizer<TEXT, IMAGE, VIDEO, AUDIO>;
REGISTER(MiniMaxM2VL, minimaxm2vl_tokenizer)
