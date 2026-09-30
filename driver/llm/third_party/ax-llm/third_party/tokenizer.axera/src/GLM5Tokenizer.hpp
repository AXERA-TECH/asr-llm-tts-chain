#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class GLM5Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string image_start_token = "<|begin_of_image|>";
    std::string image_end_token = "<|end_of_image|>";
    std::string video_start_token = "<|begin_of_video|>";
    std::string video_end_token = "<|end_of_video|>";
    std::string audio_start_token = "<|begin_of_audio|>";
    std::string audio_end_token = "<|end_of_audio|>";

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
        
        // Add prefix tokens for GLM-5
        text << "[gMASK]<sop>";

        // Find the last user message index for thinking logic
        int last_user_index = -1;
        for (size_t i = 0; i < contents.size(); i++)
        {
            if (contents[i].role == USER)
            {
                last_user_index = static_cast<int>(i);
            }
        }

        for (size_t i = 0; i < contents.size(); i++)
        {
            const auto &content = contents[i];
            
            if (content.role == SYSTEM)
            {
                text << "<|system|>" << content.data;
            }
            else if (content.role == USER)
            {
                switch (content.type)
                {
                case TEXT:
                    text << "<|user|>" << content.data;
                    break;
                case IMAGE:
                    text << "<|user|>" << image_start_token;
                    for (int j = 0; j < content.num_media * content.num_media_tokens; j++)
                    {
                        text << "<|image_pad|>";
                    }
                    text << image_end_token << content.data;
                    break;
                case VIDEO:
                    text << "<|user|>" << video_start_token;
                    for (int j = 0; j < content.num_media * content.num_media_tokens; j++)
                    {
                        text << "<|video_pad|>";
                    }
                    text << video_end_token << content.data;
                    break;
                case AUDIO:
                    text << "<|user|>" << audio_start_token;
                    for (int j = 0; j < content.num_media * content.num_media_tokens; j++)
                    {
                        text << "<|audio_pad|>";
                    }
                    text << audio_end_token << content.data;
                    break;
                default:
                    break;
                }
            }
            else if (content.role == ASSISTANT)
            {
                text << "<|assistant|>";
                
                // Handle thinking content
                std::string reasoning_content;
                std::string actual_content = content.data;
                
                // Extract reasoning content from <think> tags if present
                size_t think_start = content.data.find("<think>");
                size_t think_end = content.data.find("</think>");
                
                if (think_start != std::string::npos && think_end != std::string::npos && think_end > think_start)
                {
                    reasoning_content = content.data.substr(think_start + 7, think_end - think_start - 7);
                    // Trim leading/trailing whitespace from reasoning
                    size_t start = reasoning_content.find_first_not_of(" \n\r\t");
                    size_t end = reasoning_content.find_last_not_of(" \n\r\t");
                    if (start != std::string::npos && end != std::string::npos)
                    {
                        reasoning_content = reasoning_content.substr(start, end - start + 1);
                    }
                    // Get content after </think>
                    actual_content = content.data.substr(think_end + 8);
                    size_t content_start = actual_content.find_first_not_of(" \n\r\t");
                    if (content_start != std::string::npos)
                    {
                        actual_content = actual_content.substr(content_start);
                    }
                }
                
                // Determine if we should show thinking content
                // For messages after last user, or if think_in_prompt is true, show thinking
                bool show_thinking = (static_cast<int>(i) > last_user_index) || this->think_in_prompt;
                
                if (show_thinking && !reasoning_content.empty())
                {
                    text << "<think>" << reasoning_content << "</think>";
                }
                else
                {
                    text << "</think>";
                }
                
                if (!actual_content.empty())
                {
                    text << actual_content;
                }
            }
        }

        // Add generation prompt
        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|assistant|>";
            // By default, start with <think> for generation
            text << "<think>";
        }

        return text.str();
    }
};

// Register tokenizer variants - all use the same underlying tokenizer file
using glm5_tokenizer = GLM5Tokenizer<TEXT>;
REGISTER(GLM5, glm5_tokenizer)

using glm5vl_tokenizer = GLM5Tokenizer<TEXT, IMAGE, VIDEO>;
REGISTER(GLM5VL, glm5vl_tokenizer)
