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
