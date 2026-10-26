#!/usr/bin/env bash
# Reproducible full-video pilot. Run --selection-only first to inspect the cohort.
# Paths and service settings can be overridden via environment variables.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SAMPLE_SIZE="${SAMPLE_SIZE:-10}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-data/m3_runs_yuanjian/m3bench_random${SAMPLE_SIZE}_seed${SEED}}"
CONFIG="${CONFIG:-data/ogmem_test_lyj.resolved.yaml}"
ANNOTATIONS="${ANNOTATIONS:-data/m3/annotations/robot.json}"
VIDEOS_ROOT="${VIDEOS_ROOT:-data/m3/videos}"
# false matches the successful 0918 experiment. Set true for an LLM that
# cannot accept input_audio; ASR text will still be used.
NO_AUDIO="${NO_AUDIO:-false}"
# Only ASR/VAD currently have input/configuration-validated cache receipts.
# Keep the historical full-refresh default until dependent stages are audited.
ASR_VAD_MODE="${ASR_VAD_MODE:-refresh}"
case "$ASR_VAD_MODE" in
    refresh) asr_vad_force=true ;;
    cached) asr_vad_force=false ;;
    *) echo "ASR_VAD_MODE must be refresh or cached" >&2; exit 2 ;;
esac

extra=()
if [[ -n "${CV_SERVICE_URL:-}" ]]; then
    # Explicit legacy override only; otherwise honor cv_service in --config.
    extra+=(--set "detect_faces.params.cv_service_url=$CV_SERVICE_URL")
fi
if [[ -n "${EXCLUDE:-}" ]]; then
    extra+=(--exclude "$EXCLUDE")
fi
# Optional NPU-hosted OpenAI-compatible Whisper endpoint. When unset, keep
# the same ASR backend as the full pipeline; do not guess server endpoints.
if [[ -n "${ASR_BASE_URL:-}" ]]; then
    extra+=(--set voice_asr.params.backend=openai_http
            --set "voice_asr.params.asr_base_url=$ASR_BASE_URL")
fi

mkdir -p "$RUN_ROOT"
# Completed videos resume at the outer driver. For any video that runs,
# refresh evidence rather than reusing the old graph (ASR_VAD_MODE=cached
# permits validated ASR/VAD reuse only). Pass --force to explicitly rerun
# completed videos as well; it does not override ASR_VAD_MODE.
# Avoid verbose by default: debug logs include large base64 audio payloads.
uv run python scripts/run_m3_bench_subset.py \
    --config "$CONFIG" \
    --annotations "$ANNOTATIONS" \
    --videos-root "$VIDEOS_ROOT" \
    --sample-size "$SAMPLE_SIZE" --seed "$SEED" \
    --seconds 0 \
    --graph-store sql \
    --work-dir-root "$RUN_ROOT" \
    --pipeline config/m3/pipelines/video_episodic_full.yaml \
    --no-smart-resume \
    --qa-max-iterations 6 --qa-timeout 180 --qa-top-k 8 \
    --set qa.params.temperature=0 \
    --set qa.params.candidate_k=20 \
    --set qa.params.transcript_top_k=20 \
    --set qa.params.transcript_fusion=rrf \
    --set qa.params.context_k=8 \
    --set qa.params.include_neighbors=1 \
    --set qa.params.temporal_policy=current_clip \
    --set episodic_memory.params.force=true \
    --set "episodic_memory.params.no_audio=$NO_AUDIO" \
    --set detect_faces.params.backend=http \
    --set detect_faces.params.force=true \
    --set "voice_asr.params.force=$asr_vad_force" \
    --set speaker_diarization.params.force=true \
    --set speaker_embedding.params.force=true \
    --set "voice_segmentation.params.force=$asr_vad_force" \
    --judge ${extra[@]+"${extra[@]}"} "$@" 2>&1 | tee -a "$RUN_ROOT/driver.log"
