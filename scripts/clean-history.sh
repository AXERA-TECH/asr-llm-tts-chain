#!/usr/bin/env bash
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG:-${SDK_DIR}/config/sdk-config.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "配置文件不存在：${CONFIG_PATH}" >&2
    exit 1
fi

# The board cleanup path is intentionally dependency-free. sdk-config.yaml is
# a restricted flat mapping, so extract output files/directories (including
# Qwen3 and Hojo) and the persistent CampPlus speaker table path in awk.
PATHS_TEXT="$(awk '
  /^[[:space:]]*mp3_dir:[[:space:]]*/ {v=$0; sub(/^[^:]*:[[:space:]]*/,"",v); print v}
  /^[[:space:]]*temp_dir:[[:space:]]*/ {v=$0; sub(/^[^:]*:[[:space:]]*/,"",v); print v}
  /^[[:space:]]*text_file:[[:space:]]*/ {v=$0; sub(/^[^:]*:[[:space:]]*/,"",v); print v}
  /^[[:space:]]*output_file:[[:space:]]*/ {v=$0; sub(/^[^:]*:[[:space:]]*/,"",v); print v}
  /^[[:space:]]*output_dir:[[:space:]]*/ {v=$0; sub(/^[^:]*:[[:space:]]*/,"",v); print v}
  /^[[:space:]]*speaker_table_path:[[:space:]]*/ {v=$0; sub(/^[^:]*:[[:space:]]*/,"",v); print v}
' "${CONFIG_PATH}" | sed -e "s/^['\"]//;s/['\"]$//" | while IFS= read -r path; do
  [[ "${path}" = /* ]] || path="${SDK_DIR}/${path}"
  case "${path}" in ~/*) path="${HOME}/${path#~/}";; esac
  readlink -m -- "${path}"
done)"

# CampPlus detailed clustering logs use a fixed path so they remain separate
# from transcript output while still participating in history cleanup.
PATHS_TEXT+=$'\n/tmp/ax-audio-sdk/campplus.log'

mapfile -t HISTORY_PATHS <<< "${PATHS_TEXT}"

for path in "${HISTORY_PATHS[@]}"; do
    # Never follow a configured symlink: cleaning a symlink target could remove
    # unrelated data outside the configured cache location.
    if [[ -L "${path}" ]]; then
        echo "跳过符号链接（为安全起见）：${path}" >&2
        continue
    fi

    if [[ "${path}" == "/" || "${path}" == "${SDK_DIR}" ]]; then
        echo "拒绝清理危险路径：${path}" >&2
        exit 1
    fi

    if [[ -d "${path}" ]]; then
        # Keep cache directories in place; the audio pipeline can immediately
        # recreate/use them after cleanup.
        find "${path}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
        echo "已清理目录：${path}"
    elif [[ -e "${path}" ]]; then
        rm -f -- "${path}"
        echo "已删除文件：${path}"
    else
        echo "不存在，跳过：${path}"
    fi
done

# Hojo's optional --keep-prompt debug mode uses this fixed, narrowly scoped
# mkstemps pattern outside output_dir.
find /tmp -mindepth 1 -maxdepth 1 -type f -name 'hojo_prompt_*.bin' \
    -delete -print | while IFS= read -r path; do
  echo "已删除 Hojo prompt：${path}"
done
