#!/usr/bin/env bash
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG:-${SDK_DIR}/config/sdk-config.yaml}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

case "${DRY_RUN}" in 0|1) ;; *) echo "DRY_RUN 只接受 0 或 1" >&2; exit 2;; esac
case "${FORCE}" in 0|1) ;; *) echo "FORCE 只接受 0 或 1" >&2; exit 2;; esac

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "配置文件不存在：${CONFIG_PATH}" >&2
    exit 1
fi

if [[ "${FORCE}" != "1" ]] && pgrep -x audio-pipeline >/dev/null 2>&1; then
    echo "检测到 audio-pipeline 正在运行；请先停止链路，或明确使用 FORCE=1。" >&2
    exit 2
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

mapfile -t HISTORY_PATHS < <(printf '%s\n' "${PATHS_TEXT}" | sed '/^[[:space:]]*$/d' | sort -u)

remove_file() {
    local path="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[预览] 将删除文件：${path}"
    else
        rm -f -- "${path}"
        echo "已删除文件：${path}"
    fi
}

clear_directory() {
    local path="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        find "${path}" -mindepth 1 -print | sed 's/^/[预览] 将删除：/'
    else
        find "${path}" -mindepth 1 -depth -delete
        echo "已清理目录：${path}"
    fi
}

for path in "${HISTORY_PATHS[@]}"; do
    # Never follow a configured symlink: cleaning a symlink target could remove
    # unrelated data outside the configured cache location.
    if [[ -L "${path}" ]]; then
        echo "跳过符号链接（为安全起见）：${path}" >&2
        continue
    fi

    case "${path}" in
      /|/tmp|/var|/var/tmp|/root|/home|/data|"${SDK_DIR}")
        echo "拒绝清理危险路径：${path}" >&2
        exit 1
        ;;
    esac

    if [[ -d "${path}" ]]; then
        # Keep configured directories in place so the pipeline can reuse them.
        clear_directory "${path}"
    elif [[ -e "${path}" ]]; then
        remove_file "${path}"
    else
        echo "不存在，跳过：${path}"
    fi
done

# Hojo's optional --keep-prompt debug mode uses this fixed, narrowly scoped
# mkstemps pattern outside output_dir.
while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    remove_file "${path}"
done < <(find /tmp -mindepth 1 -maxdepth 1 -type f -name 'hojo_prompt_*.bin' -print)
