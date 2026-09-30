#pragma once
#include <sstream>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

class HunYuanTokenizer : public BaseMixinTokenizer<TEXT>
{
protected:
    const std::string bos_token = "<｜hy_begin▁of▁sentence｜>";
    const std::string user_token = "<｜hy_User｜>";
    const std::string assistant_token = "<｜hy_Assistant｜>";
    const std::string placeholder_2 = "<｜hy_place▁holder▁no▁2｜>";
    const std::string placeholder_3 = "<｜hy_place▁holder▁no▁3｜>";
    const std::string placeholder_8 = "<｜hy_place▁holder▁no▁8｜>";

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
        size_t start_index = 0;

        if (!contents.empty() && contents[0].role == SYSTEM)
        {
            text << bos_token << contents[0].data << placeholder_3;
            start_index = 1;
        }
        else
        {
            text << bos_token;
        }

        for (size_t i = start_index; i < contents.size(); ++i)
        {
            const auto &content = contents[i];
            if (content.role == USER)
            {
                text << user_token << content.data;
            }
            else if (content.role == ASSISTANT)
            {
                text << assistant_token << content.data << placeholder_2;
            }
        }

        if (!contents.empty() && contents.back().role == USER && add_generation_prompt)
        {
            text << assistant_token;
        }
        else
        {
            text << placeholder_8;
        }

        return text.str();
    }
};

REGISTER(HunYuan, HunYuanTokenizer)
