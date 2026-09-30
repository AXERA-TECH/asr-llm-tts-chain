#!/usr/bin/env bash
set -euo pipefail
SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SDK_DIR}"
export LD_LIBRARY_PATH="${SDK_DIR}/build:${SDK_DIR}/driver/apm/lib:${SDK_DIR}/driver/vad/lib:${SDK_DIR}/driver/asr/sensevoice/lib:${SDK_DIR}/driver/asr/lib:/soc/lib:/opt/lib:${LD_LIBRARY_PATH:-}"
# AX-LLM uses OpenMP internally; one worker avoids competing CPU teams beside
# the realtime APM/VAD threads while token decoding is active.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# SenseVoice resources and its AX650 front-end dependencies live together.
export AX_SENSEVOICE_SHARE_DIR="${AX_SENSEVOICE_SHARE_DIR:-${SDK_DIR}/driver/asr/sensevoice}"
CONFIG_PATH="${CONFIG:-${SDK_DIR}/config/sdk-config.yaml}"
RUNTIME_UI_ENABLED="${RUNTIME_UI_ENABLED:-1}"
RUNTIME_UI_HOST="${RUNTIME_UI_HOST:-0.0.0.0}"
RUNTIME_UI_PORT="${RUNTIME_UI_PORT:-7862}"
RUNTIME_UI_REFRESH_SECONDS="${RUNTIME_UI_REFRESH_SECONDS:-1}"
echo "Starting AX650 audio pipeline with ${CONFIG_PATH}" >&2
PIPELINE_ARGS=(--config "${CONFIG_PATH}")

RUNTIME_UI_PID=""
PIPELINE_PID=""

stop_process_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  kill -TERM -- "-${pid}" 2>/dev/null || true
  for _ in {1..50}; do
    kill -0 -- "-${pid}" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - INT TERM EXIT
  if [[ -n "${PIPELINE_PID}" ]]; then
    kill -TERM "${PIPELINE_PID}" 2>/dev/null || true
    wait "${PIPELINE_PID}" 2>/dev/null || true
  fi
  stop_process_group "${RUNTIME_UI_PID}"
  [[ -z "${RUNTIME_UI_PID}" ]] || wait "${RUNTIME_UI_PID}" 2>/dev/null || true
  return "${status}"
}

forward_signal() {
  local signal="$1"
  [[ -z "${PIPELINE_PID}" ]] || kill -"${signal}" "${PIPELINE_PID}" 2>/dev/null || true
}

trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
trap cleanup EXIT

case "${RUNTIME_UI_ENABLED}" in
  1|true|TRUE|yes|YES)
    if ! command -v uv >/dev/null 2>&1; then
      echo "无法启动运行状态页面：未找到 uv；可设置 RUNTIME_UI_ENABLED=0 仅启动主管线。" >&2
      exit 2
    fi
    setsid uv run python "${SDK_DIR}/framework/configui/runtime-ui.py" \
      --config "${CONFIG_PATH}" --base-dir "${SDK_DIR}" \
      --host "${RUNTIME_UI_HOST}" --port "${RUNTIME_UI_PORT}" \
      --refresh-seconds "${RUNTIME_UI_REFRESH_SECONDS}" &
    RUNTIME_UI_PID=$!
    echo "Runtime Gradio UI: http://${RUNTIME_UI_HOST}:${RUNTIME_UI_PORT}" >&2
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "RUNTIME_UI_ENABLED 只接受 1/0、true/false 或 yes/no。" >&2
    exit 2
    ;;
esac

build/audio-pipeline "${PIPELINE_ARGS[@]}" "$@" &
PIPELINE_PID=$!
set +e
wait "${PIPELINE_PID}"
PIPELINE_STATUS=$?
set -e
PIPELINE_PID=""
exit "${PIPELINE_STATUS}"
