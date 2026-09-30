#!/usr/bin/env bash
set -euo pipefail
SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
MAIN_PORT="${MAIN_PORT:-7860}"
APM_PORT="${APM_PORT:-7861}"
CONFIG="${CONFIG:-${SDK_DIR}/config/sdk-config.yaml}"
port_available() {
  PORT_TO_CHECK="$1" python3 - <<'PY'
import os
import socket
port = int(os.environ["PORT_TO_CHECK"])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        raise SystemExit(1)
PY
}

if port_available "${APM_PORT}"; then
  # Run each service in its own process group.  `uv run` can spawn a Python
  # child, so stopping only the uv process would leave the child listening on
  # the port after this script exits.
  setsid uv run python "${SDK_DIR}/framework/configui/apm-config-ui.py" --config "${CONFIG}" --host "${HOST}" --port "${APM_PORT}" &
  APM_PID=$!
else
  echo "APM 配置页面端口 ${APM_PORT} 已被占用，保留现有实例；如需新实例请设置 APM_PORT。" >&2
  APM_PID=""
fi

stop_process_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0

  # A process group is used so that uv and any Python descendants are all
  # stopped together.  Ignore races with processes that already exited.
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
  stop_process_group "${MAIN_PID:-}"
  stop_process_group "${APM_PID:-}"
  wait "${MAIN_PID:-}" 2>/dev/null || true
  wait "${APM_PID:-}" 2>/dev/null || true
  return "${status}"
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

if ! port_available "${MAIN_PORT}"; then
  echo "主配置页面端口 ${MAIN_PORT} 已被占用，请设置 MAIN_PORT，例如 MAIN_PORT=7870。" >&2
  exit 2
fi

setsid uv run python "${SDK_DIR}/framework/configui/sdk-config-ui.py" \
  --config "${CONFIG}" --host "${HOST}" --port "${MAIN_PORT}" \
  --apm-port "${APM_PORT}" &
MAIN_PID=$!
set +e
wait "${MAIN_PID}"
MAIN_STATUS=$?
set -e
exit "${MAIN_STATUS}"
