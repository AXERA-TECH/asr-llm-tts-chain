#!/usr/bin/env bash
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AX650_SDK_ROOT="${AX650_SDK_ROOT:-}"
CXX="${CXX:-aarch64-linux-gnu-g++}"
if [[ -z "${AX650_SDK_ROOT}" ]]; then
  echo "Set AX650_SDK_ROOT to a board SDK containing include/ and lib/." >&2
  exit 2
fi

"${CXX}" -std=c++17 -O3 -fPIC -shared \
  -I"${SDK_DIR}/driver/ns/include" -I"${SDK_DIR}/driver/ns/src" \
  -I"${AX650_SDK_ROOT}/include" \
  "${SDK_DIR}/driver/ns/src/fastenhance_c_api.cpp" \
  "${SDK_DIR}/driver/ns/src/fastenhancer.cpp" \
  -L"${AX650_SDK_ROOT}/lib" -lax_sys -lax_engine -lax_interpreter -lm \
  -Wl,-soname,libfastenhance.so -o "${SDK_DIR}/build/libfastenhance.so"
echo "Built ${SDK_DIR}/build/libfastenhance.so"
