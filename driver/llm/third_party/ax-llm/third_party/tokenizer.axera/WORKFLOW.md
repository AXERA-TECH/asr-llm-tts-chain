# Tokenizer 项目工作流指南

本文档描述了如何为新的 HuggingFace 模型添加 C++ Tokenizer 支持的完整流程。

## 项目概述

本项目将 HuggingFace Transformers 的 Tokenizer 导出为纯文本格式，并在 C++ 中实现对话模板（chat template）的拼接逻辑。

### 核心组件

```
项目结构:
├── include/BaseTokenizer.hpp      # 基类定义和 ModelType 枚举
├── src/
│   ├── BaseTokenizer.cpp          # Tokenizer 工厂
│   ├── BaseMixinTokenizer.hpp     # Mixin 基类（编码/解码实现）
│   ├── Qwen3Tokenizer.hpp         # 现有实现示例
│   └── [YourNewTokenizer].hpp     # 你的新实现
├── tests/
│   ├── convert_tokenizer.py       # 导出 HF tokenizer 为 txt
│   ├── verify_tokenizer.py        # 验证脚本（Python vs C++）
│   └── test_tokenizer.cpp         # C++ 测试程序
└── tests/assets/
    └── [model]_tokenizer.txt      # 导出的 tokenizer 文件
```

### ModelType 枚举

当前支持的模型类型（在 `include/BaseTokenizer.hpp` 中定义）：

```cpp
enum ModelType
{
    Qwen2_5,      // 0
    Qwen3,        // 1
    Qwen3VL,      // 2
    InternVL3,    // 3
    HunYuan,      // 4
    SmolVLM2,     // 5
    FastVLM,      // 6
    MiniCPM4,     // 7
    MiniCPMV4,    // 8
    Gemma3,       // 9
    Gemma3VL,     // 10
    SmolLM2,      // 11
    SmolLM3,      // 12
    GLM5,         // 13
    GLM5VL,       // 14
    KimiK25,      // 15
    KimiK25VL,    // 16
    Qwen3Omni,    // 17
    Qwen3_5,      // 18
    Qwen3_5VL,    // 19
	    MiniMaxM2,    // 20
	    MiniMaxM2VL,  // 21
	    MiniCPMO4_5,  // 22
	    InternVL3_5,  // 23
	    PaddleOCRVL,  // 24
	    Gemma4,       // 25
	    // 新增模型在这里添加...
	};
```

---

## 添加新模型的完整流程

### 步骤 1: 导出 Tokenizer

**命令:**
```bash
cd tests
python convert_tokenizer.py \
    --tokenizer_path <hf_model_path> \
    --dst_path assets/<model_name>_tokenizer.txt
```

**示例:**
```bash
python convert_tokenizer.py \
    --tokenizer_path openbmb/MiniCPM4-0.5B \
    --dst_path assets/minicpm4_tokenizer.txt
```

**验证导出成功:**
```bash
ls -lh assets/<model_name>_tokenizer.txt
```

---

### 步骤 2: 分析对话模板

**目的:** 了解模型如何拼接对话消息为文本。

**Python 分析脚本:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("<hf_model_path>", trust_remote_code=True)

# 1. 查看 chat template
print("Chat template:")
print(tokenizer.chat_template)

# 2. 测试对话拼接
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！很高兴见到你。"}
]

text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
print("\nApplied template:")
print(repr(text))

# 3. 查看 tokenization 结果
print("\nTokenized:", tokenizer.encode(text))

# 4. 查看特殊 token
print("\nSpecial tokens:", tokenizer.special_tokens_map)
print("EOS:", tokenizer.eos_token)
print("BOS:", tokenizer.bos_token)
```

**关键观察点:**
- 消息格式: `<|im_start|>system\n{content}<|im_end|>\n` (如 Qwen 格式)
- 角色名称: `user`/`assistant` 或 `user`/`model` (如 Gemma)
- 系统消息处理: 独立消息还是拼接到首个 user
- 生成提示: `add_generation_prompt=True` 时追加什么
- 特殊 token: BOS/EOS 的位置

**常见模板模式:**

| 模型系列 | 格式 | 示例 |
|---------|------|------|
| Qwen/MiniCPM | `<|im_start|>{role}\n{content}<|im_end|>\n` | `<|im_start|>system\n...<|im_end|>\n` |
| Gemma | `<bos><start_of_turn>{role}\n{content}<end_of_turn>\n` | `<start_of_turn>user\n...<end_of_turn>\n` |
| Llama | `<|begin_of_text|>...<|start_header_id|>...` | `<|start_header_id|>user<|end_header_id|>...` |
| SmolLM | 复杂 Metadata + system | `## Metadata\nKnowledge Cutoff...` |

---

### 步骤 3: 实现 C++ Tokenizer 类

**创建文件:** `src/[YourModel]Tokenizer.hpp`

**基础模板（纯文本模型）:**
```cpp
#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

class YourModelTokenizer : public BaseMixinTokenizer<TEXT>
{
public:
    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        // 检查内容类型
        for (const auto &content : contents)
        {
            if (!this->support(content.type))
            {
                ALOGE("unsupport content type: %d", content.type);
                return {};
            }
        }

        std::stringstream text;
        
        // TODO: 添加 BOS token（如果需要）
        // text << "<s>";
        
        for (const auto &content : contents)
        {
            if (content.role == SYSTEM)
            {
                // TODO: 实现 system 消息格式
                text << "<|im_start|>system\n"
                     << content.data << "<|im_end|>\n";
            }
            else if (content.role == USER)
            {
                // TODO: 实现 user 消息格式
                text << "<|im_start|>user\n"
                     << content.data << "<|im_end|>\n";
            }
            else if (content.role == ASSISTANT)
            {
                // 处理 think_in_prompt 选项
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

        // 添加生成提示
        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n";
        }

        return text.str();
    }
};

// 注册 tokenizer
REGISTER(YourModel, YourModelTokenizer)
```

**多模态模型模板（支持图片/视频）:**
```cpp
// 使用模板参数指定支持的内容类型
template <ContentType... Types>
class YourMultimodalTokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string image_pad_token = "<image>";  // 根据模型修改

public:
    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        std::stringstream text;
        
        for (const auto &content : contents)
        {
            if (content.role == USER)
            {
                switch (content.type)
                {
                case TEXT:
                    text << "<|im_start|>user\n"
                         << content.data << "<|im_end|>\n";
                    break;
                case IMAGE:
                    // 处理图片输入
                    text << "<|im_start|>user\n"
                         << image_pad_token << "\n"  // 图片占位符
                         << content.data << "<|im_end|>\n";
                    break;
                case VIDEO:
                    // 处理视频输入（类似图片）
                    break;
                default:
                    break;
                }
            }
            // ... 其他角色处理
        }
        
        return text.str();
    }
};

// 注册不同版本的 tokenizer
using yourmodel_tokenizer = YourMultimodalTokenizer<TEXT>;
REGISTER(YourModel, yourmodel_tokenizer)

using yourmodelvl_tokenizer = YourMultimodalTokenizer<TEXT, IMAGE>;
REGISTER(YourModelVL, yourmodelvl_tokenizer)
```

**参考实现:**
- 简单 IM 格式: `src/Qwen3Tokenizer.hpp`
- Gemma 格式: `src/Gemma3Tokenizer.hpp`
- 复杂 System: `src/SmolLM3Tokenizer.hpp`
- 多模态: `src/MiniCPM4Tokenizer.hpp`

---

### 步骤 4: 注册到工厂

**1. 添加 ModelType 枚举（include/BaseTokenizer.hpp）:**
```cpp
enum ModelType
{
    // ... 现有类型
    
    YourModel,    // 新增
    YourModelVL,  // 新增（如果有视觉版本）
};
```

**2. 包含头文件并注册（src/BaseTokenizer.cpp）:**
```cpp
#include "YourModelTokenizer.hpp"

// 不需要额外代码，REGISTER 宏会自动注册
```

---

### 步骤 5: 编译项目

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

**检查编译成功:**
```bash
ls -l test_tokenizer
```

---

### 步骤 6: 验证实现

**使用通用验证脚本:**
```bash
cd tests
python verify_tokenizer.py \
    --model <hf_model_path> \
    --model-type <model_type_num> \
    --tokenizer-path assets/<model_name>_tokenizer.txt \
    [--test-image]  # 如果是多模态模型
```

**示例:**
```bash
# 纯文本模型
python verify_tokenizer.py \
    --model openbmb/MiniCPM4-0.5B \
    --model-type 7

# 多模态模型
python verify_tokenizer.py \
    --model openbmb/MiniCPM-V-4 \
    --model-type 8 \
    --test-image \
    --image-placeholder '<image>'
```

**手动测试（C++ 程序）:**
```bash
cd build
./test_tokenizer \
    -t ../tests/assets/<model_name>_tokenizer.txt \
    -m <model_type_num>
```

**验证要点:**
1. Python 和 C++ 生成的文本格式完全一致
2. Token IDs 序列完全一致
3. Decode 后的文本与原始文本一致
4. 多轮对话处理正确
5. 多模态占位符处理正确（如果是 VL 模型）

---

## 快速参考

### 一键添加新模型脚本

```bash
#!/bin/bash
# add_new_model.sh <hf_model_path> <model_type_name>

HF_PATH=$1
MODEL_NAME=$2

# 转换为小写文件名
FILE_NAME=$(echo $MODEL_NAME | tr '[:upper:]' '[:lower:]')
ASSET_NAME=$(echo $HF_PATH | tr '/' '_' | tr '-' '_')

echo "=== 步骤 1: 导出 Tokenizer ==="
cd tests
python convert_tokenizer.py \
    --tokenizer_path $HF_PATH \
    --dst_path assets/${ASSET_NAME}_tokenizer.txt

echo "=== 步骤 2: 分析对话模板 ==="
python3 << EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("$HF_PATH", trust_remote_code=True)
print("Chat template:", tokenizer.chat_template[:200] if tokenizer.chat_template else "None")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Example output:", repr(text))
EOF

echo "=== 步骤 3: 请手动实现 src/${MODEL_NAME}Tokenizer.hpp ==="
echo "=== 步骤 4: 请更新 include/BaseTokenizer.hpp 和 src/BaseTokenizer.cpp ==="
echo "=== 步骤 5: 编译项目 ==="
cd ../build && make -j4

echo "=== 步骤 6: 验证 ==="
cd ../tests
python verify_tokenizer.py \
    --model $HF_PATH \
    --model-type <NUM>  # 需要手动指定 ModelType 数值
```

### 常见对话模板模式

**模式 1: IM 风格（Qwen/MiniCPM）**
```cpp
text << "<|im_start|>" << role << "\n"
     << content << "<|im_end|>\n";
```

**模式 2: Turn 风格（Gemma）**
```cpp
// BOS 在开头
text << "<bos>";
// System 拼接到首个 user
if (first_user) text << system_content << "\n\n";
// 每个消息
text << "<start_of_turn>" << role << "\n"
     << content << "<end_of_turn>\n";
```

**模式 3: Header 风格（Llama）**
```cpp
text << "<|start_header_id|>" << role << "<|end_header_id|>\n\n"
     << content << "<|eot_id|>";
```

---

## 调试技巧

### 1. 对比文本输出

Python:
```python
print(repr(text))
```

C++:
```cpp
printf("text: %s\n", text.c_str());
// 或者查看原始字符
for (char c : text) printf("%d ", (unsigned char)c);
```

### 2. 检查特殊字符

常见问题:
- `\n` vs `\r\n`
- 末尾缺少换行
- 多余空格
- BOS/EOS 位置

### 3. Token ID 不匹配

Python:
```python
ids = tokenizer.encode(text)
for i, id in enumerate(ids[:20]):
    print(f"{i}: {id} -> {repr(tokenizer.decode([id]))}")
```

### 4. 使用 GDB 调试 C++

```bash
gdb --args ./test_tokenizer -t tokenizer.txt -m 7
(gdb) break YourModelTokenizer::apply_chat_template
(gdb) run
```

---

## 提交前检查清单

- [ ] Tokenizer 文件已导出到 `tests/assets/`
- [ ] C++ 类已实现并注册
- [ ] ModelType 枚举已更新
- [ ] 编译无警告无错误
- [ ] Python vs C++ 文本输出一致
- [ ] Python vs C++ Token IDs 完全一致
- [ ] Decode 测试通过
- [ ] 多轮对话测试通过
- [ ] 多模态测试通过（如果是 VL 模型）
- [ ] 代码遵循现有风格

---

## 扩展阅读

- [HuggingFace Chat Templating](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Jinja2 模板语法](https://jinja.palletsprojects.com/)
- 项目参考: [mnn-llm](https://github.com/wangzhaode/mnn-llm), [llm-export](https://github.com/wangzhaode/llm-export)

### SmolLM2 (新增)
- **ModelType**: 11
- **格式**: IM 格式 `<|im_start|>{role}
{content}<|im_end|>
`
- **特点**: 
  - BOS: `<|im_start|>` (ID: 1)
  - EOS: `<|im_end|>` (ID: 2)
  - 与 Qwen/MiniCPM 格式兼容
- **文件**: `src/SmolLM2Tokenizer.hpp`

### GLM-5 (新增)
- **ModelType**: 13 (GLM5), 14 (GLM5VL)
- **Tokenizer 文件**: `glm5_tokenizer.txt` (两个版本共用同一个文件)
- **格式**: `[gMASK]<sop><|system|>{content}<|user|>{content}<|assistant|>{content}`
- **特点**:
  - 前缀: `[gMASK]<sop>`
  - System: `<|system|>{content}`
  - User: `<|user|>{content}`
  - Assistant: `<|assistant|>{content}`
  - 支持 thinking: `<think>{reasoning}</think>` 或 `</think>`
  - 多模态: `<|begin_of_image|>`, `<|end_of_image|>`, `<|begin_of_video|>`, `<|end_of_video|>`
- **实现**: 使用模板类 `GLM5Tokenizer<TEXT>` 和 `GLM5Tokenizer<TEXT, IMAGE, VIDEO>`
- **文件**: `src/GLM5Tokenizer.hpp`

### Kimi-K2.5 (新增)
- **ModelType**: 15 (KimiK25), 16 (KimiK25VL)
- **Tokenizer 文件**: `kimi_k25_tokenizer.txt` (两个版本共用同一个文件)
- **格式**: `<|im_system|>system<|im_middle|>{content}<|im_end|><|im_user|>user<|im_middle|>{content}<|im_end|><|im_assistant|>assistant<|im_middle|<think></think>{content}<|im_end|>`
- **特点**:
  - System: `<|im_system|>system<|im_middle|>{content}<|im_end|>`
  - User: `<|im_user|>user<|im_middle|>{content}<|im_end|>`
  - Assistant: `<|im_assistant|>assistant<|im_middle|<think></think>{content}<|im_end|>`
  - 支持图片: `<|media_begin|>image<|media_content|><|media_pad|><|media_end|>`
  - 支持视频: `<|kimi_k25_video_placeholder|>`
- **实现**: 使用模板类 `KimiK25Tokenizer<TEXT>` 和 `KimiK25Tokenizer<TEXT, IMAGE, VIDEO>`
- **文件**: `src/KimiK25Tokenizer.hpp`

### Qwen3-Omni (新增)
- **ModelType**: 17 (Qwen3Omni)
- **Tokenizer 文件**: `qwen3_omni_tokenizer.txt`
- **格式**: 与 Qwen3 相同 `<|im_start|>{role}\n{content}<|im_end|>\n`
- **特点**:
  - 与 Qwen3VL 相同格式
  - 支持文本、图像、视频、音频
  - Audio: `<|audio_start|>...<|audio_pad|>...<|audio_end|>`
- **实现**: 复用 `Qwen3Tokenizer<TEXT, IMAGE, VIDEO, AUDIO>`
- **文件**: `src/Qwen3Tokenizer.hpp`

### MiniMax-M2.1 (新增)
- **ModelType**: 20 (MiniMaxM2), 21 (MiniMaxM2VL)
- **Tokenizer 文件**: `minimaxm2_tokenizer.txt` (两个版本共用同一个文件)
- **格式**: `]~!b[]~b]system\n...{content}[e~[\n]~b]user\n{content}[e~[\n]~b]ai\n{content}[e~[\n`
- **特点**:
  - BOS: `]~!b[` (ID: 200034)
  - EOS: `[e~[` (ID: 200020)
  - System: `]~!b[]~b]system\n{content}[e~[\n`
  - User: `]~b]user\n{content}[e~[\n`
  - Assistant: `]~b]ai\n{content}[e~[\n`
  - 支持图片: `]<]image[>[`
  - 支持视频: `]<]video[>[`
  - 支持语音: `]<]speech[>[`
- **实现**: 使用模板类 `MiniMaxM2Tokenizer<TEXT>` 和 `MiniMaxM2Tokenizer<TEXT, IMAGE, VIDEO, AUDIO>`
- **文件**: `src/MiniMaxM2Tokenizer.hpp`

### MiniCPM-o-4_5 (新增)
- **ModelType**: 22 (MiniCPMO4_5)
- **Tokenizer 文件**: `minicpmo4_5_tokenizer.txt`
- **格式**: 与 MiniCPM4 相同 `<s><|im_start|>{role}\n{content}<|im_end|>\n`
- **特点**:
  - 与 MiniCPM4 相同格式
  - 支持文本、图像、音频
  - Audio: `<|audio_start|>...<|audio|>...<|audio_end|>`
- **实现**: 复用 `MiniCPM4Tokenizer<TEXT, IMAGE, AUDIO>`
- **文件**: `src/MiniCPM4Tokenizer.hpp`


### Qwen3.5 (新增)
- **ModelType**: 18 (Qwen3_5), 19 (Qwen3_5VL)
- **Tokenizer 文件**: `qwen3_5_tokenizer.txt`
- **格式**: 与 Qwen3 相同 `<|im_start|>{role}\n{content}<|im_end|>\n`
- **特点**:
  - Vocab 大小: 248077 (vs Qwen3-Omni 的 151676)
  - Qwen3_5: 文本版本（预留）
  - Qwen3_5VL: 支持文本、图像、视频（不含 AUDIO）
  - Special token IDs: 248044-248076
- **实现**: 复用 `Qwen3Tokenizer<TEXT>` 和 `Qwen3Tokenizer<TEXT, IMAGE, VIDEO>`
- **文件**: `src/Qwen3Tokenizer.hpp`, `tests/assets/qwen3_5_tokenizer.txt`

### InternVL3.5 (新增)
- **ModelType**: 23 (InternVL3_5)
- **Tokenizer 文件**: `internvl3_5_tokenizer.txt`（1B/2B 可共用）
- **格式**: 与 InternVL3 相同 `<|im_start|>{role}\n{content}<|im_end|>\n`
- **特点**:
  - 支持文本、图像、视频
  - 图片 token: `<img><IMG_CONTEXT>... </img>`
  - 视频 token: 与图片占位符格式一致
- **实现**: 复用 `InternVL3Tokenizer`，注册 `InternVL3_5`
- **文件**: `src/InternVL3Tokenizer.hpp`


### 更新说明 (2024-02-13)
- Qwen3.5 改为 Qwen3_5VL (ModelType 19)
- Qwen3_5 (ModelType 18) 预留用于后续纯文本版本
- Qwen3_5VL 支持 TEXT, IMAGE, VIDEO (不含 AUDIO)
