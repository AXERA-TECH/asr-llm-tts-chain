#pragma once
#include <sstream>
#include <algorithm>
#include <ctime>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

class SmolLM3Tokenizer : public BaseMixinTokenizer<TEXT>
{
protected:
    // Get current date in format like "10 February 2026"
    static std::string get_current_date()
    {
        std::time_t now = std::time(nullptr);
        std::tm *tm_info = std::localtime(&now);
        
        const char *months[] = {"January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December"};
        
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%d %s %d", tm_info->tm_mday, months[tm_info->tm_mon], tm_info->tm_year + 1900);
        return std::string(buffer);
    }
    
    // Build default system message with metadata
    static std::string build_default_system_message(const std::string &custom_instructions = "")
    {
        std::stringstream ss;
        ss << "## Metadata\n\n";
        ss << "Knowledge Cutoff Date: June 2025\n";
        ss << "Today Date: " << get_current_date() << "\n";
        ss << "Reasoning Mode: /think\n\n";
        ss << "## Custom Instructions\n\n";
        
        if (!custom_instructions.empty())
        {
            ss << custom_instructions << "\n\n";
        }
        else
        {
            ss << "You are a helpful AI assistant named SmolLM, trained by Hugging Face. ";
            ss << "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process ";
            ss << "before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle ";
            ss << "of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop ";
            ss << "well-considered thinking process. Please structure your response into two main sections: Thought and Solution ";
            ss << "using the specified format: <think> Thought section </think> Solution section. In the Thought section, ";
            ss << "detail your reasoning process in steps. Each step should include detailed considerations such as analysing ";
            ss << "questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, ";
            ss << "refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, ";
            ss << "explorations, and reflections from the Thought section, systematically present the final solution that you ";
            ss << "deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed ";
            ss << "to reach the conclusion.\n\n";
        }
        
        return ss.str();
    }

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
        
        // System message is always at the beginning
        text << "<|im_start|>system\n";
        
        // Extract system content if present
        std::string system_content;
        size_t start_idx = 0;
        bool has_system = false;
        
        if (!contents.empty() && contents[0].role == SYSTEM)
        {
            system_content = contents[0].data;
            start_idx = 1;
            has_system = true;
        }
        
        if (has_system && system_content.find("/system_override") != std::string::npos)
        {
            // Use system content directly, removing the /system_override tag
            std::string cleaned = system_content;
            size_t pos = cleaned.find("/system_override");
            if (pos != std::string::npos)
            {
                cleaned.erase(pos, 16); // length of "/system_override"
            }
            // Trim trailing whitespace
            cleaned.erase(cleaned.find_last_not_of(" \n\r\t") + 1);
            text << cleaned << "<|im_end|>\n";
        }
        else
        {
            // Use default system message with optional custom instructions
            text << build_default_system_message(system_content);
            text << "<|im_end|>\n";
        }
        
        // Process remaining messages
        for (size_t i = start_idx; i < contents.size(); i++)
        {
            const auto &content = contents[i];
            
            if (content.role == USER)
            {
                text << "<|im_start|>user\n"
                     << content.data << "<|im_end|>\n";
            }
            else if (content.role == ASSISTANT)
            {
                // SmolLM3 always includes <think> tags in assistant responses
                // The actual thinking content should be in the data if present
                text << "<|im_start|>assistant\n";
                
                // If content doesn't start with <think>, add empty think tags
                if (content.data.find("<think>") != 0)
                {
                    text << "<think>\n\n</think>\n";
                }
                
                // Remove leading newlines from content
                std::string cleaned_data = content.data;
                size_t pos = 0;
                while (pos < cleaned_data.size() && (cleaned_data[pos] == '\n' || cleaned_data[pos] == '\r'))
                {
                    pos++;
                }
                if (pos > 0)
                {
                    cleaned_data = cleaned_data.substr(pos);
                }
                
                text << cleaned_data << "<|im_end|>\n";
            }
        }

        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n";
            // Note: for /no_think mode, it would add "<think>\n\n</think>\n" here
            // We default to /think mode which doesn't add the empty think block
        }

        return text.str();
    }
};

REGISTER(SmolLM3, SmolLM3Tokenizer)
