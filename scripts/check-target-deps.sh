#!/usr/bin/env bash
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
errors=0
runtime_files=(
    driver/apm/lib/libwebrtc-audio-processing-2.so.1
    driver/vad/lib/libten_vad.so
    driver/asr/sensevoice/lib/libax_asr_api.so
    build/run-alsa-apm
    build/audio-pipeline
    build/libfastenhance.so
    build/qwen3-worker
    build/hojo-tts-resident
    config/sdk-config.yaml
    config/default-config.yaml
    framework/configui/runtime-ui.py
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
    models-for-asr-chain/asr/sensevoice/sensevoice.axmodel
    models-for-asr-chain/asr/sensevoice/tokens.txt
    models-for-asr-chain/asr/sensevoice/am.mvn
    models-for-asr-chain/asr/sensevoice/chn_jpn_yue_eng_ko_spectok.bpe.model
)
for index in {0..27}; do
    runtime_files+=("models-for-asr-chain/llm/qwen3-1.7b/qwen3_p128_l${index}_together.axmodel")
done
for index in {0..9}; do
    runtime_files+=("models-for-asr-chain/tts/hojo/models/lm_s8/qwen3_p8_l${index}_together.axmodel")
done

for relative_path in "${runtime_files[@]}"; do
    file="${SDK_DIR}/${relative_path}"
    if [[ -r "${file}" ]]; then
        echo "OK: ${relative_path}"
    else
        echo "MISSING: ${relative_path}" >&2
        errors=$((errors + 1))
    fi
done

if command -v uv >/dev/null 2>&1; then
    echo "OK: uv ($(command -v uv))"
else
    echo "MISSING: uv（run-target.sh 默认需要它启动 Gradio 运行状态页面）" >&2
    errors=$((errors + 1))
fi

if command -v aplay >/dev/null 2>&1; then
    echo "OK: aplay ($(command -v aplay))"
else
    echo "WARN: aplay is not installed (required only for optional monitoring)"
fi
if command -v lame >/dev/null 2>&1; then
    echo "OK: lame ($(command -v lame))"
elif command -v ffmpeg >/dev/null 2>&1 && \
     ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libmp3lame; then
    echo "OK: ffmpeg has libmp3lame ($(command -v ffmpeg))"
else
    echo "WARN: no MP3 encoder; install lame or FFmpeg with libmp3lame"
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libasound.so'; then
    echo "WARN: libasound.so is not visible to ldconfig; verify target ALSA runtime" >&2
fi

(( errors == 0 ))
