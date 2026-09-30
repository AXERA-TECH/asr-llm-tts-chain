#!/usr/bin/env bash
# 整体编译 AX650 实时音频主程序及其运行时组件 (aarch64-linux)
# 目标平台配置见 config/aarch64-linux.ini
# 预编译库位于 driver/apm，链接时只指定库的搜索路径，
# 不设置 rpath，运行时在目标平台通过 LD_LIBRARY_PATH 指定库的路径。

set -euo pipefail

# SDK 根目录和产物目录（绝对路径）
SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${SDK_DIR}/build"

# 交叉编译工具链（与 config/aarch64-linux.ini 保持一致）
CXX="${CXX:-aarch64-linux-gnu-g++}"
CC="${CC:-${CXX%++}cc}"
CXX_PATH="$(command -v "${CXX}" || true)"
CC_PATH="$(command -v "${CC}" || true)"
if [[ -z "${CXX_PATH}" || -z "${CC_PATH}" ]]; then
    echo "Cross compiler not found: CC=${CC}, CXX=${CXX}" >&2
    exit 1
fi
CROSS_SYSROOT="${CROSS_SYSROOT:-}"
CROSS_FLAGS=()
CMAKE_SYSROOT_ARGS=()
if [[ -n "${CROSS_SYSROOT}" ]]; then
    if [[ ! -d "${CROSS_SYSROOT}" ]]; then
        echo "CROSS_SYSROOT is not a directory: ${CROSS_SYSROOT}" >&2
        exit 1
    fi
    CROSS_SYSROOT="$(cd "${CROSS_SYSROOT}" && pwd)"
    CROSS_FLAGS+=("--sysroot=${CROSS_SYSROOT}")
    CMAKE_SYSROOT_ARGS+=("-DCMAKE_SYSROOT=${CROSS_SYSROOT}")
fi

# 头文件与库路径
APM_DIR="${SDK_DIR}/driver/apm"
INCLUDE_DIRS=(
    "-I${APM_DIR}/include"
    "-I${APM_DIR}/include/webrtc-audio-processing-2"
)
LIB_DIR="${APM_DIR}/lib"
ALSA_SYSROOT="${ALSA_SYSROOT:-${CROSS_SYSROOT}}"
ALSA_INCLUDE_DIR="${ALSA_INCLUDE_DIR:-}"
ALSA_LIBRARY="${ALSA_LIBRARY:-}"
mkdir -p "${BUILD_DIR}"

AX650_SDK_ROOT="${AX650_SDK_ROOT:-}"
CAMPPLUS_SRC="${SDK_DIR}/driver/campplus"
CAMPPLUS_EXTRA=()
CAMPPLUS_LIBS=()
if [[ -d "${AX650_SDK_ROOT}/include" && -d "${AX650_SDK_ROOT}/lib" ]]; then
    # Build the small static fbank dependency once in the normal build tree.
    # The two archives are vendored under driver/campplus/third_party, so a
    # board build does not require network access.
    KNF_ROOT="${BUILD_DIR}/campplus-third_party"
    if [[ ! -f "${KNF_ROOT}/kaldi-native-fbank-1.22.3/CMakeLists.txt" ]]; then
        mkdir -p "${KNF_ROOT}"
        tar -xzf "${CAMPPLUS_SRC}/third_party/kaldi-native-fbank-1.22.3.tar.gz" -C "${KNF_ROOT}"
        cp "${CAMPPLUS_SRC}/third_party/kissfft-febd4caeed32e33ad8b2e0bb5ea77542c40f18ec.zip" "${KNF_ROOT}/"
    fi
    cmake -S "${KNF_ROOT}/kaldi-native-fbank-1.22.3" -B "${BUILD_DIR}/campplus-knf" \
        -DCMAKE_CXX_COMPILER="${CXX}" -DCMAKE_C_COMPILER="${CC}" \
        "${CMAKE_SYSROOT_ARGS[@]}" \
        -DBUILD_SHARED_LIBS=OFF -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF \
        -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF -DKALDI_NATIVE_FBANK_ENABLE_CHECK=OFF >/dev/null
    cmake --build "${BUILD_DIR}/campplus-knf" --target kaldi-native-fbank-core -j2
    CAMPPLUS_EXTRA+=("-DCAMPPLUS_AVAILABLE=1" "-I${CAMPPLUS_SRC}/include" "-I${KNF_ROOT}/kaldi-native-fbank-1.22.3" "-I${AX650_SDK_ROOT}/include")
    CAMPPLUS_LIBS+=("${BUILD_DIR}/campplus-knf/lib/libkaldi-native-fbank-core.a" "${BUILD_DIR}/campplus-knf/lib/libkissfft-float.a" "-L${AX650_SDK_ROOT}/lib" "-lax_sys" "-lax_engine" "-lax_interpreter")
else
    CAMPPLUS_EXTRA+=("-I${CAMPPLUS_SRC}/include")
fi

# Native C++ runtime for the target board.  It parses sdk-config.yaml and
# owns the complete audio/VAD/ASR/MP3 pipeline; Python is only used by the
# optional Gradio configuration pages.
"${CXX}" "${CROSS_FLAGS[@]}" -std=c++17 -O2 \
    -I"${SDK_DIR}/framework" -I"${SDK_DIR}/framework/configui" \
    -I"${SDK_DIR}/driver/ns/include" -I"${SDK_DIR}/driver/vad/include" -I"${SDK_DIR}/driver/asr/sensevoice/include" "${CAMPPLUS_EXTRA[@]}" \
    "${SDK_DIR}/framework/audio_pipeline.cpp" \
    "${CAMPPLUS_SRC}/src/campplus.cpp" \
    "${SDK_DIR}/framework/configui/apm-config.cpp" \
    "${CAMPPLUS_LIBS[@]}" -ldl -pthread -Wl,--export-dynamic -Wl,--allow-shlib-undefined \
    -o "${BUILD_DIR}/audio-pipeline"
echo "Build done: ${BUILD_DIR}/audio-pipeline"

if [[ -n "${ALSA_SYSROOT}" && -d "${ALSA_SYSROOT}" ]]; then
    if [[ -z "${ALSA_INCLUDE_DIR}" ]]; then
        for candidate in "${ALSA_SYSROOT}/usr/include" "${ALSA_SYSROOT}/include"; do
            if [[ -r "${candidate}/alsa/asoundlib.h" ]]; then
                ALSA_INCLUDE_DIR="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${ALSA_LIBRARY}" ]]; then
        for candidate in \
            "${ALSA_SYSROOT}/usr/lib/aarch64-linux-gnu/libasound.so" \
            "${ALSA_SYSROOT}/lib/aarch64-linux-gnu/libasound.so" \
            "${ALSA_SYSROOT}/usr/lib/libasound.so" \
            "${ALSA_SYSROOT}/lib/libasound.so"; do
            if [[ -r "${candidate}" ]]; then
                ALSA_LIBRARY="${candidate}"
                break
            fi
        done
    fi
fi
ALSA_INCLUDE_DIR="${ALSA_INCLUDE_DIR:-/usr/include}"
ALSA_LIBRARY="${ALSA_LIBRARY:-/usr/lib/aarch64-linux-gnu/libasound.so}"
if [[ -r "${ALSA_INCLUDE_DIR}/alsa/asoundlib.h" && -r "${ALSA_LIBRARY}" ]]; then
    "${CXX}" "${CROSS_FLAGS[@]}" -std=c++17 -O2 \
        "${INCLUDE_DIRS[@]}" -I"${ALSA_INCLUDE_DIR}" \
        -I"${SDK_DIR}/driver/alsa_briage" -I"${SDK_DIR}/framework" -I"${SDK_DIR}/framework/configui" \
        -DWEBRTC_LIBRARY_IMPL -DWEBRTC_POSIX \
        "${SDK_DIR}/driver/alsa_briage/run-alsa-apm.cpp" \
        "${SDK_DIR}/driver/alsa_briage/alsa-apm-backend.cpp" \
        "${SDK_DIR}/framework/configui/apm-config.cpp" \
        -L"${LIB_DIR}" -lwebrtc-audio-processing-2 "${ALSA_LIBRARY}" -lrt -ldl -pthread \
        -Wl,--allow-shlib-undefined \
        -o "${BUILD_DIR}/run-alsa-apm"
    echo "Build done: ${BUILD_DIR}/run-alsa-apm"
else
    echo "ALSA headers/library not found; skipped run-alsa-apm." >&2
    echo "Set ALSA_INCLUDE_DIR and ALSA_LIBRARY, or ALSA_SYSROOT." >&2
fi

if [[ -d "${AX650_SDK_ROOT}/include" && -d "${AX650_SDK_ROOT}/lib" ]]; then
    AX650_SDK_ROOT="${AX650_SDK_ROOT}" CROSS_SYSROOT="${CROSS_SYSROOT}" CXX="${CXX}" \
        "${SDK_DIR}/driver/ns/build-fastenhance.sh"

    # Qwen3 runs as a persistent subprocess.  Keeping it in a separate
    # executable isolates the AX-LLM lifetime from the realtime audio path.
    QWEN_BUILD_DIR="${BUILD_DIR}/qwen3-worker-cmake"
    cmake -S "${SDK_DIR}/driver/llm" -B "${QWEN_BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${SDK_DIR}/driver/llm/toolchains/ax650-aarch64.cmake" \
        -DCMAKE_C_COMPILER="${CC_PATH}" -DCMAKE_CXX_COMPILER="${CXX_PATH}" \
        "${CMAKE_SYSROOT_ARGS[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${BUILD_DIR}" \
        -DAX650_SDK_DIR="${AX650_SDK_ROOT}"
    cmake --build "${QWEN_BUILD_DIR}" --target qwen3-worker -j2
    echo "Build done: ${BUILD_DIR}/qwen3-worker"

    HOJO_BUILD_DIR="${BUILD_DIR}/hojo-tts-cmake"
    cmake -S "${SDK_DIR}/driver/tts/hojo" -B "${HOJO_BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${SDK_DIR}/driver/tts/hojo/toolchains/ax650-aarch64.cmake" \
        -DCMAKE_C_COMPILER="${CC_PATH}" -DCMAKE_CXX_COMPILER="${CXX_PATH}" \
        "${CMAKE_SYSROOT_ARGS[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${BUILD_DIR}" \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${BUILD_DIR}" \
        -DAX650_SDK_DIR="${AX650_SDK_ROOT}"
    cmake --build "${HOJO_BUILD_DIR}" --target hojo-tts-resident -j2
    echo "Build done: ${BUILD_DIR}/hojo-tts-resident"
else
    echo "AX650_SDK_ROOT is unavailable; skipped AX650 runtime components." >&2
fi
echo "在目标平台运行时请指定库的路径，例如："
echo "  ./scripts/run-target.sh"
