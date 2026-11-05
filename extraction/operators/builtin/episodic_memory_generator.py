"""Episodic memory generator — multimodal LLM operator.

Joins per-clip ``IMAGE`` frame children + an optional ``AUDIO`` child
of a ``VIDEO`` ``RawNode`` into one multimodal LLM call and returns
structured episodic memory text (``description`` + ``characters`` +
``events`` + ``entities``).

The operator stays pure: it does not touch the graph or filesystem
beyond reading child ``RawNode.content`` bytes. The driver is
responsible for looking up children, persisting the resulting
``ExtractedNode``, and materialising any downstream edges.

Output ``extracted_content`` shape (matches the schema below)::

    {
      "description": "The robot arm picks up a red cup and places it on a tray.",
      "characters": [{"name": "robot_arm", "role": "agent"}],
      "events": [{"type": "manipulation", "summary": "Pick and place cup."}],
      "entities": [{"name": "red cup", "type": "object"}],
      "frame_count": 5,
      "has_audio": true,
    }

Used by ``m3.episodic_memory`` driver against an LLM provider that
accepts ``image_url`` + ``input_audio`` content parts (MiniMax-M3 via
``providers.llm.minimax_llm.MiniMaxLLM``, or any OpenAI-compatible
multimodal backend).
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from core.graph_enums import RawDataType
from extraction.operators.base import (
    ExtractionResult,
    OperatorInfo,
    default_info,
)

if TYPE_CHECKING:
    from core.graph_models import RawNode
    from core.interfaces import LLM
    from core.models import RequestContext


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema (also embedded in the prompt by the LLM provider)
# ---------------------------------------------------------------------------

EPISODIC_MEMORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": (
                "A 2-4 sentence natural-language description of what happens "
                "in the clip. Reference visible characters, actions, and "
                "objects. If audio is present, mention any audible cues "
                "(speech, sound effects) that complement the visual."
            ),
        },
        "characters": {
            "type": "array",
            "description": "People / agents / characters that appear in the clip.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "events": {
            "type": "array",
            "description": "Discrete events / actions occurring in the clip.",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "summary": {"type": "string"},
                    "ts_sec": {"type": "number"},
                },
                "required": ["type", "summary"],
            },
        },
        "entities": {
            "type": "array",
            "description": "Objects / places / named entities referenced in the clip.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
}


_SYSTEM_PROMPT = (
    "You are an episodic memory generator for long-video question answering. "
    "Given sampled frames (and optionally the clip's audio), produce a "
    "structured JSON summary of what happens in this clip.\n\n"
    "Guidance:\n"
    "- description: ALWAYS 2-4 sentences, present tense. Cover the "
    "characters, their actions, the setting, and the objects in the "
    "scene. When a transcript block is provided, also weave the spoken "
    "content (who says what, and any reported emotions or plans) into "
    "the description so QA can answer from speech as well as vision. "
    "Do NOT compress the visual description — both visual and speech "
    "content are required.\n"
    "- VERBATIM SPEECH RULE (additive, do not replace the visual "
    "description): when the transcript contains short, load-bearing "
    "phrases (emotional states like 'I'm exhausted' or 'so happy', "
    "plans like 'put them in an album', requests like 'fetch the "
    "pictures in my bag'), include those EXACT phrases in the "
    "description. Do NOT paraphrase them away into generic words like "
    "'happiness' or 'arranging' — the answerer scores verbatim against "
    "the ground truth, and losing the load-bearing word loses the "
    "match. Surface the verbatim phrase in addition to the visual "
    "description, not instead of it.\n"
    "- characters: list every distinct agent or person you can identify. "
    "Use the same descriptive label consistently within a single video "
    "(e.g. 'Woman in white hoodie' rather than 'a woman' in one clip and "
    "'Woman 1' in another). Do NOT invent proper names that are not "
    "supported by either the visual content or the speech transcript. "
    "When a transcript clearly attributes speech to a name, surface that "
    "name in the characters list and in the description.\n"
    "- events: list each observable action with the rough timestamp "
    "(in seconds from clip start) when possible.\n"
    "- Track visible object moves in chronological order: distinguish where "
    "an object starts, where it is moved, and its last observed location. "
    "Preserve concrete spatial relations (inside, on, hanging from) rather "
    "than replacing them with 'near'. Do not infer a destination when the "
    "object disappears from view or confuse adjacent objects.\n"
    "- A spoken name may identify the listener, not the speaker. Do not "
    "assign unnamed speech or first-person pronouns to a visible person "
    "without evidence; keep uncertain attribution explicitly unknown.\n"
    "- entities: list objects / places / things mentioned in the clip.\n"
    "- If frames are silent and the audio adds nothing, omit characters / "
    "events rather than fabricate.\n"
)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class EpisodicMemoryGenerator:
    """Multimodal LLM operator: clip frames (+ audio) -> episodic memory.

    Configuration (kwargs)::

        llm: providers.llm.LLM-protocol-conformant instance (MiniMaxLLM,
            OpenAILLM, or MockLLM for tests). Required.
        max_frames: int — cap on frames included in the prompt. The clip
            often has dozens of frames; sending all of them is wasteful and
            most models cap input images anyway. Default 8.
        image_detail: "low" | "high" | "auto" — passed through to OpenAI's
            image_url content part. Default "low" to bound token cost.
        include_audio: bool — when True (default), include the clip's
            AUDIO child if present. Set False to skip audio even if
            available (cheaper, vision-only path).
        temperature: float — sampling temperature. Default 0.2 (low
            for stable structured output).
        use_video: bool — when True, send the WHOLE clip as one
            ``video_url`` content part (re-encoded to an LLM-friendly
            mp4 via ffmpeg, cached as a sidecar) instead of sampled
            ``image_url`` frames. The video carries the audio track in
            the file, so the ``input_audio`` part is NOT attached in
            this mode (some backends — e.g. Volcengine Ark — reject
            ``input_audio`` for video-capable models). When the clip's
            ``source_path`` is missing or the re-encode fails, the
            operator falls back to frame mode. Default False.
        video_fps: int — target FPS for the re-encoded video. Default 10.
        video_width: int — target pixel width for the re-encoded video
            (height scales proportionally). Default 512.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        max_frames: int = 8,
        image_detail: str = "low",
        include_audio: bool = True,
        temperature: float = 0.2,
        use_video: bool = False,
        video_fps: int = 10,
        video_width: int = 512,
    ) -> None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        if image_detail not in ("low", "high", "auto"):
            raise ValueError(
                f"image_detail must be low|high|auto, got {image_detail!r}",
            )
        if video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {video_fps}")
        if video_width <= 0:
            raise ValueError(f"video_width must be positive, got {video_width}")
        self._llm = llm
        self._max_frames = int(max_frames)
        self._image_detail = image_detail
        self._include_audio = bool(include_audio)
        self._temperature = float(temperature)
        self._use_video = bool(use_video)
        self._video_fps = int(video_fps)
        self._video_width = int(video_width)

    @property
    def supported_types(self) -> list[RawDataType]:
        return [RawDataType.VIDEO]

    @property
    def extractor_type(self) -> str:
        return "episodic_memory_generator"

    @property
    def schema_name(self) -> str:
        return "episodic_memory_v1"

    @property
    def info(self) -> OperatorInfo:
        info = default_info(self)
        info.description = (
            "Multimodal LLM operator: takes a VIDEO node's frame + audio "
            "children, returns structured episodic memory text "
            "(description / characters / events / entities). Output is "
            "stored as the parent's extracted_content."
        )
        info.config_schema = {
            "type": "object",
            "properties": {
                "max_frames": {"type": "integer", "minimum": 1, "maximum": 64},
                "image_detail": {"enum": ["low", "high", "auto"]},
                "include_audio": {"type": "boolean"},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "use_video": {"type": "boolean"},
                "video_fps": {"type": "integer", "minimum": 1, "maximum": 30},
                "video_width": {"type": "integer", "minimum": 64, "maximum": 1920},
            },
        }
        # The operator consumes its own structured output (no graph
        # children); future graph layer can link to a SemanticMemory
        # node if semantic_memory_consolidator runs over multiple
        # episodic outputs.
        info.materialization = None
        return info

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def extract(
        self,
        raw_node: RawNode,
        ctx: RequestContext,  # type: ignore[reportUnusedArgument]  # part of the operator contract
        *,
        frame_nodes: list[RawNode] | None = None,
        audio_node: RawNode | None = None,
        asr_segments: list[dict] | None = None,
    ) -> ExtractionResult:
        """Generate episodic memory for ``raw_node`` (the VIDEO clip).

        Args:
            raw_node: The VIDEO RawNode whose metadata identifies the clip
                (``video_id``, ``clip_id`` are read for context only; the
                actual data comes from ``frame_nodes`` + ``audio_node``).
            frame_nodes: Optional pre-fetched list of IMAGE RawNode
                children. If ``None``, the operator does NOT auto-fetch
                — drivers must pass frames explicitly to keep this
                operator pure (graph access lives in the driver).
            audio_node: Optional pre-fetched AUDIO RawNode.
            asr_segments: Optional list of ASR segment dicts
                (``{start_sec, end_sec, text, speaker_id?, words[]}``)
                produced by the ``voice_asr`` operator for the same
                clip's audio. When present, the operator injects
                them as a timestamped transcript block in the user
                prompt so the LLM grounds its description in the
                actual speech. When absent, no transcript block is
                included.
        """
        frames = list(frame_nodes or [])

        # Video mode: send the whole clip as one video_url part. The
        # audio track rides inside the file, so no input_audio part is
        # attached (backends like Volcengine Ark reject input_audio
        # for video-capable models). Any failure falls back to frames.
        video_part = None
        if self._use_video:
            source_path = (raw_node.metadata or {}).get("source_path")
            if source_path:
                try:
                    video_part = _video_part_from_source(
                        source_path,
                        fps=self._video_fps,
                        width=self._video_width,
                        include_audio=self._include_audio,
                    )
                except Exception as exc:
                    logger.warning(
                        "video re-encode failed for %s (%s: %s) — "
                        "falling back to frame mode",
                        raw_node.node_id, type(exc).__name__, exc,
                    )
            else:
                logger.warning(
                    "use_video set but %s has no metadata.source_path "
                    "— falling back to frame mode",
                    raw_node.node_id,
                )

        if video_part is None and not frames:
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": "no frames provided",
                    "description": "",
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error="episodic_memory_generator requires frame_nodes",
            )

        # Subsample frames evenly if there are more than max_frames.
        # (Skipped in video mode — frames are not sent at all.)
        if video_part is None and len(frames) > self._max_frames:
            indices = _evenly_spaced_indices(len(frames), self._max_frames)
            frames = [frames[i] for i in indices]

        messages = self._build_messages(
            raw_node, frames, audio_node,
            asr_segments=normalise_speech_transcript(asr_segments),
            video_part=video_part,
            # Video mode implies a backend that takes whole-video input and
            # rejects input_audio parts (e.g. Volcengine Ark minimax-m3).
            # When we fall back to frames inside video mode, the audio part
            # must be skipped too — the transcript block still carries the
            # speech content.
            include_audio=(
                self._include_audio and not (self._use_video and video_part is None)
            ),
        )
        try:
            # Note: complete_multimodal uses the OpenAI-compatible
            # content shape (text + image_url + input_audio). MiniMax-M3
            # accepts this directly via the OpenAI client; OpenAI's
            # gpt-4o-audio-preview family accepts it as well.
            llm_response = self._llm.complete_multimodal(
                messages, EPISODIC_MEMORY_SCHEMA,
            )
            content = _normalise_episodic_response(dict(llm_response))
        except Exception as exc:  # network / schema / parse failure
            logger.warning(
                "episodic_memory_generator LLM call failed for %s: %s",
                raw_node.node_id, exc,
            )
            return ExtractionResult(
                node_id=raw_node.node_id,
                extracted_content={
                    "error": str(exc),
                    "description": "",
                },
                confidence=0.0,
                schema_used=self.schema_name,
                error=str(exc),
            )

        # Merge LLM response with provenance metadata. The text lives
        # at the top level so downstream consumers (semantic memory
        # consolidator, control loop retrieval) can read it directly.
        content["frame_count"] = 0 if video_part is not None else len(frames)
        content["video_input"] = video_part is not None
        if video_part is not None:
            content["video_fps"] = self._video_fps
            content["video_width"] = self._video_width
            # The audio track is embedded in the video file we sent.
            content["audio_attached"] = self._include_audio
        else:
            content["audio_attached"] = any(
                part.get("type") == "input_audio"
                for message in messages if isinstance(message.get("content"), list)
                for part in message["content"]
            )
        content["speech_transcript"] = normalise_speech_transcript(asr_segments)
        content["has_transcript"] = bool(content["speech_transcript"])
        content["has_audio"] = content["audio_attached"] or content["has_transcript"]
        if raw_node.metadata is not None:
            for src_key in ("video_id", "clip_id", "start_sec", "duration_sec"):
                if src_key in raw_node.metadata:
                    content[f"source_{src_key}"] = raw_node.metadata[src_key]

        return ExtractionResult(
            node_id=raw_node.node_id,
            extracted_content=content,
            confidence=0.8,  # heuristic; could become per-field self-reported
            schema_used=self.schema_name,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        raw_node: RawNode,
        frames: list[RawNode],
        audio_node: RawNode | None,
        asr_segments: list[dict] | None = None,
        video_part: dict | None = None,
        include_audio: bool | None = None,
    ) -> list[dict]:
        """Construct the multimodal chat message list for the LLM call.

        ``include_audio`` overrides the constructor's ``include_audio``
        for this call (None → constructor value). Used by the video-mode
        frames fallback to skip the ``input_audio`` part on backends
        that reject it.
        """
        # User prompt: text describing the task + clip metadata + the
        # actual frames (and optionally audio + ASR transcript).
        user_text_parts: list[str] = [
            ("Describe what happens in this video clip based on the "
             "provided video. Output JSON only.")
            if video_part is not None else
            ("Describe what happens in this video clip based on the "
             "provided frames and audio (if any). Output JSON only."),
        ]
        meta = raw_node.metadata or {}
        if "video_id" in meta:
            user_text_parts.append(f"video_id: {meta['video_id']}")
        if "clip_id" in meta:
            user_text_parts.append(f"clip_id: {meta['clip_id']}")
        if "duration_sec" in meta:
            user_text_parts.append(
                f"clip duration: {meta['duration_sec']}s",
            )

        # ASR transcript block (text-only) — works regardless of
        # whether the LLM accepts input_audio content parts. When
        # ``voice_asr`` has been run on this clip, the driver passes
        # its per-segment records here. When speaker_diarization has
        # also run, each segment dict carries a ``speaker_id`` we
        # surface so the LLM can attribute dialogue to characters.
        if asr_segments:
            transcript_lines: list[str] = []
            for seg in asr_segments:
                start = float(seg.get("start_sec", 0.0))
                end = float(seg.get("end_sec", 0.0))
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                speaker_id = seg.get("speaker_id")
                if speaker_id:
                    transcript_lines.append(
                        f"[{start:6.2f}-{end:6.2f}s] {speaker_id}: {text}"
                    )
                else:
                    transcript_lines.append(
                        f"[{start:6.2f}-{end:6.2f}s] {text}"
                    )
            if transcript_lines:
                user_text_parts.append(
                    "Speech transcript (timestamps in seconds; speaker "
                    "labels appear when diarization is available):\n"
                    + "\n".join(transcript_lines)
                )

        content_parts: list[dict] = [
            {"type": "text", "text": "\n".join(user_text_parts)},
        ]
        if video_part is not None:
            # Whole-clip video: frames and the separate audio part are
            # both superseded — the audio track is embedded in the file.
            content_parts.append(video_part)
            return [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content_parts},
            ]
        for frame in frames:
            image_part = _image_part_from_node(frame, self._image_detail)
            if image_part is not None:
                content_parts.append(image_part)
        if (self._include_audio if include_audio is None else include_audio) \
                and audio_node is not None:
            audio_part = _audio_part_from_node(audio_node)
            if audio_part is not None:
                content_parts.append(audio_part)

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]


# ---------------------------------------------------------------------------
# Response / content-part helpers
# ---------------------------------------------------------------------------

def normalise_speech_transcript(segments: list[dict] | None) -> list[dict]:
    """Canonical evidence shared by persistence and cache validation."""
    return [
        {"start_sec": float(seg.get("start_sec", 0.0) or 0.0),
         "end_sec": float(seg.get("end_sec", 0.0) or 0.0),
         "text": str(seg.get("text", "")).strip(),
         "speaker_id": seg.get("speaker_id")}
        for seg in (segments or [])
        if isinstance(seg, dict) and str(seg.get("text", "")).strip()
    ]


def _normalise_episodic_response(response: dict) -> dict:
    """Normalize provider-specific field drift into our canonical schema.

    Real multimodal models don't always honour a supplied JSON schema.
    MiniMax-M3, for example, may return ``summary`` or ``text`` instead
    of ``description`` even after the JSON-only fallback. Keep the
    provider layer permissive and normalise here, where the operator
    knows its output contract.
    """
    content = dict(response)
    # A person/entity sub-object must not masquerade as an episodic memory.
    if "name" in content and not any(
        isinstance(content.get(key), list) for key in ("characters", "events", "entities")
    ):
        raise ValueError("Invalid episodic memory: received a person/entity fragment")
    if not content.get("description"):
        for key in ("text", "summary", "memory", "caption"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                content["description"] = val.strip()
                break
    if not content.get("description"):
        raise ValueError("Invalid episodic memory: missing non-empty description")
    if not isinstance(content["description"], str) or not content["description"].strip():
        raise ValueError("Invalid episodic memory: description must be a non-empty string")

    for key in ("characters", "events", "entities"):
        if key in content and not isinstance(content[key], list):
            raise ValueError(f"Invalid episodic memory: {key} must be a list")
        content.setdefault(key, [])
    return content


def _image_part_from_node(frame: RawNode, detail: str) -> dict | None:
    """Build an ``image_url`` content part from an IMAGE ``RawNode``.

    Prefer ``frame.content`` bytes when present. If the graph keeps the
    image bytes on disk (the normal ``m3.extract_frames`` path), fall
    back to ``frame.metadata.source_path`` and read the JPEG from there.
    Returns None when neither source is available — skipping a frame is
    preferable to crashing the whole multimodal call.
    """
    content = frame.content
    if isinstance(content, bytes) and content:
        b64 = base64.b64encode(content).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        }
    if isinstance(content, str) and content:
        # RawNode.content may already be a data URI or URL.
        return {"type": "image_url", "image_url": {"url": content, "detail": detail}}

    source_path = (frame.metadata or {}).get("source_path")
    if isinstance(source_path, str) and source_path:
        try:
            raw = Path(source_path).read_bytes()
        except OSError:
            logger.warning("frame source_path is unreadable: %s", source_path)
            return None
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        }
    return None


def _audio_part_from_node(audio: RawNode) -> dict | None:
    """Build an ``input_audio`` content part from an AUDIO ``RawNode``.

    Reads metadata.source_path when content is empty.
    Returns None when neither source has bytes — the caller drops the
    part. Detects WAV/MP3 magic bytes and sets the format hint
    accordingly; defaults to WAV.
    """
    content = audio.content
    if not content:
        source_path = (audio.metadata or {}).get("source_path")
        if source_path:
            try:
                content = Path(source_path).read_bytes()
            except OSError:
                logger.warning("audio source_path is unreadable: %s", source_path)
                return None
    if not content:
        return None
    # ffmpeg can emit a valid, header-only WAV for a trailing video clip.
    # Nonempty container bytes do not imply that there are audio samples.
    # Keep the visual/transcript evidence, but do not send an empty waveform
    # to multimodal providers (Omni rejects it with HTTP 400).
    raw = content
    if isinstance(raw, str):
        try:
            raw = base64.b64decode(raw, validate=True)
        except (ValueError, binascii.Error):
            raw = b""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(raw), "rb") as wav:
                # Pipe-produced WAV headers use 0xffffffff for unknown
                # lengths, even when no sample bytes follow the header.
                if not wav.readframes(1):
                    logger.warning("Skipping WAV audio with no samples")
                    return None
        except (wave.Error, EOFError):
            # Do not silently discard corrupt/unsupported audio. Preserve
            # the existing provider-side validation for those payloads.
            pass
    if isinstance(content, str):
        # Pre-encoded payload (rare); pass through.
        return {
            "type": "input_audio",
            "input_audio": {"data": content, "format": "wav"},
        }
    fmt = "wav"
    if content[:3] == b"ID3" or content[:2] == b"\xff\xfb":
        fmt = "mp3"
    elif content[:4] != b"RIFF":
        # Unknown container; still send as wav and let the LLM error if
        # the model can't decode it.
        fmt = "wav"
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(content).decode("ascii"),
            "format": fmt,
        },
    }


# Sub-second re-encodes (residual trim slices) are rejected by the
# video-understanding API with an opaque 400 — fall back to frames.
_MIN_LLM_VIDEO_SECONDS = 1.0


def _prepare_llm_video(
    source_path: str,
    *,
    fps: int = 10,
    width: int = 512,
    include_audio: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    timeout_sec: float = 300.0,
) -> tuple[Path, str]:
    """Re-encode a clip into an LLM-friendly mp4; return (path, b64).

    Source clips come out of the slicer as high-bitrate stream copies
    (~27 MB for 30 s) — far too large for a base64 ``video_url`` part.
    We re-encode to ``width`` px, ``fps`` frames/s, CRF 30 (~350 KB for
    30 s at 512px/10fps) keeping a mono 16 kHz AAC track — unless
    ``include_audio`` is False, in which case the audio track is
    stripped (``-an``): ``--use-video --no-audio`` must not silently
    upload the raw audio inside the video while reporting
    ``audio_attached=False`` (review item 11). The result is cached as a
    sidecar next to the source so re-extraction (``--force``) doesn't
    pay the ffmpeg cost again, and the encode knobs — including the
    audio setting — are baked into the sidecar name so different
    settings never reuse a stale encode.
    """
    import subprocess

    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"clip source not found: {src}")
    sidecar = src.with_name(
        f"{src.stem}.llm-f{fps}-w{width}-{'a' if include_audio else 'na'}.mp4"
    )
    if not sidecar.exists():
        audio_args = (
            ["-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "32k"]
            if include_audio else ["-an"]
        )
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-i", str(src),
            "-vf", f"scale={width}:-2,fps={fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
            *audio_args,
            str(sidecar),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
        )
        if proc.returncode != 0 or not sidecar.exists():
            try:
                sidecar.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f"ffmpeg re-encode failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[:300]}"
            )
    data = sidecar.read_bytes()
    if not data:
        raise RuntimeError(f"re-encoded video is empty: {sidecar}")
    _ensure_video_stream(sidecar, ffmpeg_bin=ffmpeg_bin)
    return sidecar, base64.b64encode(data).decode("ascii")


def _ensure_video_stream(sidecar: Path, *, ffmpeg_bin: str = "ffmpeg") -> None:
    """Raise when the sidecar is unusable as LLM video input.

    Two degenerate cases burn an API call on an opaque 400:

    * No video stream at all (a streamless container, e.g. an
      audio-only re-encode).
    * A near-empty video — sources like a 0.17 s residual slice at a
      trim boundary encode to a 1-frame clip that Ark's video
      pipeline rejects. Require at least ``_MIN_LLM_VIDEO_SECONDS``.
    """
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        # ffprobe shipped with the ffmpeg apt package; if it is somehow
        # absent, skip the guard (best effort).
        return
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_type:format=duration",
         "-of", "json", str(sidecar)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        info = json.loads(probe.stdout or "{}")
    except json.JSONDecodeError:
        info = {}
    if not (info.get("streams") or []):
        raise RuntimeError(
            f"re-encoded video has no video stream: {sidecar} "
            f"(degenerate clip?)"
        )
    try:
        duration = float((info.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    if duration is not None and duration < _MIN_LLM_VIDEO_SECONDS:
        raise RuntimeError(
            f"re-encoded video is too short for LLM video input "
            f"({duration:.2f}s < {_MIN_LLM_VIDEO_SECONDS}s): {sidecar}"
        )


def _video_part_from_source(
    source_path: str,
    *,
    fps: int = 10,
    width: int = 512,
    include_audio: bool = True,
) -> dict:
    """Build a ``video_url`` content part from a clip's source path."""
    _, b64 = _prepare_llm_video(
        source_path, fps=fps, width=width, include_audio=include_audio
    )
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{b64}"},
    }


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """Pick ``k`` indices from ``[0, n)`` as evenly as possible."""
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    step = n / k
    return [int(i * step) for i in range(k)]


__all__ = [
    "EpisodicMemoryGenerator",
    "EPISODIC_MEMORY_SCHEMA",
]
