#!/usr/bin/env bash
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG:-${1:-${SDK_DIR}/config/sdk-config.yaml}}"
DEFAULT_CONFIG="${DEFAULT_CONFIG:-${2:-${SDK_DIR}/config/default-config.yaml}}"

if [[ ! -f "${DEFAULT_CONFIG}" ]]; then
    echo "默认配置不存在：${DEFAULT_CONFIG}" >&2
    exit 1
fi

mkdir -p "$(dirname "${CONFIG_PATH}")"
if [[ ! -e "${CONFIG_PATH}" ]]; then
    cp -- "${DEFAULT_CONFIG}" "${CONFIG_PATH}"
    chmod u+w "${CONFIG_PATH}"
    echo "已从默认配置创建：${CONFIG_PATH}"
    exit 0
fi
if [[ ! -f "${CONFIG_PATH}" || -L "${CONFIG_PATH}" ]]; then
    echo "配置必须是普通文件且不能是符号链接：${CONFIG_PATH}" >&2
    exit 1
fi

# Add the ASR filter and TTS backlog controls without replacing board audio,
# voice or generation settings. Only the obsolete factory assistant prompt
# is upgraded; a user-written system prompt is preserved.
temporary="$(mktemp "${CONFIG_PATH}.migrate.XXXXXX")"
trap 'rm -f "${temporary}"' EXIT
awk '
  function wanted(section, key) {
    return (section == "llm" && (key == "min_asr_characters" || key == "max_asr_punctuation_ratio")) ||
      (section == "hojo_tts" && (key == "min_chinese_characters" || key == "min_english_words" ||
        key == "backlog_threshold" || key == "notice_dir"))
  }
  function flush(section, i, key) {
    for (i = 1; i <= counts[section]; ++i) {
      key = keys[section, i]
      if (!seen[section, key]) print defaults[section, key]
    }
  }
  FNR == NR {
    if ($0 ~ /^[a-z_]+:[[:space:]]*$/) { section = $0; sub(/:.*/, "", section) }
    if ($0 ~ /^  [a-z_]+:/) {
      key = $1; sub(/:$/, "", key)
      if (wanted(section, key)) {
        keys[section, ++counts[section]] = key
        defaults[section, key] = $0
      }
      if (section == "llm" && key == "system_prompt") system_prompt = $0
    }
    next
  }
  FNR == 1 { section = "" }
  /^[a-z_]+:[[:space:]]*$/ {
    flush(section)
    section = $0; sub(/:.*/, "", section)
  }
  /^  [a-z_]+:/ {
    key = $1; sub(/:$/, "", key)
    seen[section, key] = 1
  }
  section == "llm" && /^  system_prompt: *"?你是一个乐于助人的助手。"? *$/ {
    if (system_prompt != "") { print system_prompt; next }
  }
  { print }
  END { flush(section) }
' "${DEFAULT_CONFIG}" "${CONFIG_PATH}" >"${temporary}"
if ! cmp -s "${CONFIG_PATH}" "${temporary}"; then
    backup="${CONFIG_PATH}.pre-runtime-fixes.bak"
    [[ -e "${backup}" ]] || cp -p -- "${CONFIG_PATH}" "${backup}"
    chmod --reference="${CONFIG_PATH}" "${temporary}"
    mv -f -- "${temporary}" "${CONFIG_PATH}"
    echo "已补齐 ASR/TTS 配置并更新旧版默认提示词：${CONFIG_PATH}（备份：${backup}）"
else
    rm -f -- "${temporary}"
fi
trap - EXIT

# Older deployments predate KWS. Preserve all board-specific values and only
# insert the complete top-level KWS section from the current default config.
if grep -q '^kws:[[:space:]]*$' "${CONFIG_PATH}"; then
    echo "配置无需迁移：${CONFIG_PATH}"
    exit 0
fi

backup="${CONFIG_PATH}.pre-kws.bak"
if [[ ! -e "${backup}" ]]; then
    cp -p -- "${CONFIG_PATH}" "${backup}"
fi

temporary="$(mktemp "${CONFIG_PATH}.migrate.XXXXXX")"
trap 'rm -f "${temporary}"' EXIT
awk '
  FNR == NR {
    if ($0 ~ /^kws:[[:space:]]*$/) capture = 1
    else if (capture && $0 ~ /^[^[:space:]#][^:]*:[[:space:]]*/) capture = 0
    if (capture) block = block $0 ORS
    next
  }
  !inserted && /^vad:[[:space:]]*$/ {
    if (block == "") exit 42
    printf "%s", block
    inserted = 1
  }
  { print }
  END {
    if (!inserted) {
      if (block == "") exit 42
      printf "\n%s", block
    }
  }
' "${DEFAULT_CONFIG}" "${CONFIG_PATH}" >"${temporary}"
chmod --reference="${CONFIG_PATH}" "${temporary}"
mv -f -- "${temporary}" "${CONFIG_PATH}"
trap - EXIT
echo "已补齐 KWS 配置：${CONFIG_PATH}（备份：${backup}）"
