#!/usr/bin/env bash
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-root@10.126.29.50}"
TARGET_DIR="${TARGET_DIR:-/root/wangzizhen/asr-chain}"
SSH_PASSWORD="${SSH_PASSWORD:-}"

ssh_command=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10)
if [[ -n "${SSH_PASSWORD}" ]]; then
    ssh_command=(sshpass -p "${SSH_PASSWORD}" "${ssh_command[@]}")
fi

# Only deploy files consumed by scripts/run-target.sh, the production
# ASR -> Qwen3 -> Hojo chain, and the optional Gradio configuration UI. Build
# trees, source files and test programs stay on the development host. The
# config UI is deployed with the same relative layout as the development host
# so scripts/start-ui.sh and `uv run python framework/configui/*.py` work
# unchanged on the board.
deploy_items=(
    build/run-alsa-apm
    build/audio-pipeline
    build/libfastenhance.so
    build/qwen3-worker
    build/hojo-tts-resident
    config/sdk-config.yaml
    config/default-config.yaml
    scripts/run-target.sh
    scripts/check-target-deps.sh
    scripts/clean-history.sh
    scripts/start-ui.sh
    README.md
    .python-version
    pyproject.toml
    uv.lock
    framework/configui/apm-config-ui.py
    framework/configui/sdk-config-ui.py
    framework/configui/runtime-ui.py
    driver/apm/lib/libwebrtc-audio-processing-2.so.1
    driver/vad/lib/libten_vad.so
    driver/asr/sensevoice/lib/libax_asr_api.so
    models-for-asr-chain/ns/fastenhance/fastenhance_48k.axmodel
    models-for-asr-chain/vad/ten-vad/ten-vad.axmodel
    models-for-asr-chain/campplus/campplus.axmodel
    models-for-asr-chain/llm/qwen3-1.7b/config.json
    models-for-asr-chain/llm/qwen3-1.7b/post_config.json
    models-for-asr-chain/llm/qwen3-1.7b/qwen3_tokenizer.txt
    models-for-asr-chain/llm/qwen3-1.7b/model.embed_tokens.weight.bfloat16.bin
    models-for-asr-chain/llm/qwen3-1.7b/qwen3_post.axmodel
    models-for-asr-chain/tts/hojo/models/tokenizer.json
    models-for-asr-chain/tts/hojo/models/tokenizer_config.json
    models-for-asr-chain/tts/hojo/models/fine_local.axmodel
    models-for-asr-chain/tts/hojo/models/decoder_sq.axmodel
    models-for-asr-chain/tts/hojo/models/speaker_embeds.bin
    models-for-asr-chain/tts/hojo/models/speaker_vecs.bin
    models-for-asr-chain/tts/hojo/models/id2code.bin
    models-for-asr-chain/tts/hojo/models/lm_s8/embed_tokens.bin
    models-for-asr-chain/tts/hojo/models/lm_s8/post_config.json
    models-for-asr-chain/tts/hojo/models/lm_s8/tokenizer.txt
    models-for-asr-chain/tts/hojo/models/lm_s8/qwen3_post.axmodel
    models-for-asr-chain/asr/sensevoice/tokens.txt
    models-for-asr-chain/asr/sensevoice/am.mvn
    models-for-asr-chain/asr/sensevoice/chn_jpn_yue_eng_ko_spectok.bpe.model
)
for index in {0..27}; do
    deploy_items+=("models-for-asr-chain/llm/qwen3-1.7b/qwen3_p128_l${index}_together.axmodel")
done
for index in {0..9}; do
    deploy_items+=("models-for-asr-chain/tts/hojo/models/lm_s8/qwen3_p8_l${index}_together.axmodel")
done

sensevoice_model="models-for-asr-chain/asr/sensevoice/sensevoice.axmodel"
deploy_items+=("${sensevoice_model}")

manifest_file="$(mktemp)"
changed_file="$(mktemp)"
trap 'rm -f "${manifest_file}" "${changed_file}"' EXIT

declare -A expected_sizes=()
declare -A expected_hashes=()
total_bytes=0
for item in "${deploy_items[@]}"; do
    if [[ ! -r "${SDK_DIR}/${item}" ]]; then
        echo "Missing deploy artifact: ${SDK_DIR}/${item}" >&2
        exit 1
    fi
    size="$(stat -c '%s' "${SDK_DIR}/${item}")"
    hash="$(sha256sum "${SDK_DIR}/${item}")"
    hash="${hash%% *}"
    expected_sizes["${item}"]="${size}"
    expected_hashes["${item}"]="${hash}"
    total_bytes=$((total_bytes + size))
    printf '%s %s %s\n' "${size}" "${hash}" "${item}" >>"${manifest_file}"
done

# Reuse an existing board-side model tree when upgrading from the old layout.
# This avoids retransferring multi-gigabyte model files solely because the
# project-level directory was renamed.
"${ssh_command[@]}" "${TARGET}" "
set -eu
target_dir='${TARGET_DIR}'
old_models=\"\${target_dir}/models\"
new_models=\"\${target_dir}/models-for-asr-chain\"
if [ -d \"\${old_models}\" ] && [ ! -e \"\${new_models}\" ]; then
    mv \"\${old_models}\" \"\${new_models}\"
fi
"

# Compare size first, then SHA-256. The board prints only missing or changed
# paths, so unchanged multi-gigabyte model files never cross the network.
"${ssh_command[@]}" "${TARGET}" "
set -eu
target_dir='${TARGET_DIR}'
mkdir -p \"\${target_dir}\"
while read -r expected_size expected_hash relative_path; do
    destination=\"\${target_dir}/\${relative_path}\"
    if [ ! -f \"\${destination}\" ]; then
        printf '%s\\n' \"\${relative_path}\"
        continue
    fi
    actual_size=\$(stat -c '%s' \"\${destination}\")
    if [ \"\${actual_size}\" != \"\${expected_size}\" ]; then
        printf '%s\\n' \"\${relative_path}\"
        continue
    fi
    actual_hash=\$(sha256sum \"\${destination}\")
    actual_hash=\${actual_hash%% *}
    if [ \"\${actual_hash}\" != \"\${expected_hash}\" ]; then
        printf '%s\\n' \"\${relative_path}\"
    fi
done
" <"${manifest_file}" >"${changed_file}"

mapfile -t changed_items <"${changed_file}"
changed_bytes=0
for item in "${changed_items[@]}"; do
    [[ -n "${item}" ]] || continue
    size="${expected_sizes[${item}]}"
    hash="${expected_hashes[${item}]}"
    changed_bytes=$((changed_bytes + size))
    remote_path="${TARGET_DIR}/${item}"
    remote_parent="${remote_path%/*}"
    remote_temp="${remote_path}.deploy-part"
    echo "UPLOAD: ${item} (${size} bytes)"

    # Receive into a sibling temporary file, verify it on the board, and then
    # rename it atomically. A dropped connection never corrupts the live file.
    "${ssh_command[@]}" "${TARGET}" "
set -eu
mkdir -p '${remote_parent}'
temporary='${remote_temp}'
trap 'rm -f \"\${temporary}\"' EXIT
cat >\"\${temporary}\"
actual_size=\$(stat -c '%s' \"\${temporary}\")
actual_hash=\$(sha256sum \"\${temporary}\")
actual_hash=\${actual_hash%% *}
[ \"\${actual_size}\" = '${size}' ]
[ \"\${actual_hash}\" = '${hash}' ]
mv -f \"\${temporary}\" '${remote_path}'
trap - EXIT
" <"${SDK_DIR}/${item}"
done

unchanged_count=$((${#deploy_items[@]} - ${#changed_items[@]}))
echo "Incremental deploy: ${unchanged_count} unchanged, ${#changed_items[@]} uploaded, ${changed_bytes}/${total_bytes} bytes transferred"

# Remove artifacts left by older full-tree deployments. All paths below are
# known test programs, intermediate build trees, or superseded model inputs.
"${ssh_command[@]}" "${TARGET}" "cd '${TARGET_DIR}' && \
    rm -rf \
      build/sensevoice-runner build/hojo-tts build/hojo-tts-oneshot \
      build/hojo-tts-smoke-test build/libhojo-engine-mode.so \
      build/hojo-tts-cmake build/hojo-tts-dfc4-cmake build/qwen3-worker-cmake \
      framework/hojo/test-output framework/hojo/hojo_tts_cpp_verified \
      framework/hojo/hojo_tts_oneshot.cpp framework/hojo/hojo_tts_smoke_test.cpp \
      framework/hojo/tts_driver.cpp framework/hojo/README.upstream.md \
      framework/hojo/ax-llm-core.patch models \
      models-for-asr-chain/tts/hojo/bin \
      models-for-asr-chain/tts/hojo/models/Hojo-TTS-Light-40M-voice.npz \
      models-for-asr-chain/asr/sensevoice/streaming_sensevoice.axmodel && \
    chmod +x build/run-alsa-apm build/audio-pipeline build/qwen3-worker \
      build/hojo-tts-resident scripts/check-target-deps.sh \
      scripts/run-target.sh scripts/clean-history.sh \
      scripts/start-ui.sh && \
    chmod 444 config/default-config.yaml && \
    LD_LIBRARY_PATH=driver/apm/lib:/soc/lib:/opt/lib ldd build/run-alsa-apm && \
    ./scripts/check-target-deps.sh"

echo "Deployed production runtime to ${TARGET}:${TARGET_DIR}"
