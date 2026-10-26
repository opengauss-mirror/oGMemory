"""Shared configuration for M3-Bench scripts.

Centralises paths, account identifiers, and clip defaults so individual
scripts can be invoked with sensible defaults when run from the m3-agent
working directory (where ``data/annotations/robot.json`` and
``data/videos/robot/`` live).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Tenant id used for all M3-Bench data inside oG-Memory's graph.
# All videos for a benchmark run share one account so multi-tenant
# isolation does not block retrieval across videos.
M3BENCH_ACCOUNT_ID = "m3bench"

# Default agent / user ids — operators need them for RequestContext.
M3BENCH_AGENT_ID = "m3bench-agent"
M3BENCH_USER_ID = "m3bench-user"


@dataclass(frozen=True)
class PreprocessConfig:
    """Defaults for video → clip slicing.

    Attributes:
        clip_seconds: Length of each output clip (matches M3-Agent default).
        fps: Frame sampling rate inside each clip (M3-Agent default).
        videos_root: Directory containing video files.
        clips_root: Directory where sliced clips are written.
        ffmpeg_bin: ffmpeg executable name or path.
        ffprobe_bin: ffprobe executable name or path.
        ffmpeg_log_level: ``-loglevel`` value for ffmpeg invocation.
    """
    clip_seconds: int = 30
    fps: int = 5
    videos_root: str = "data/videos"
    clips_root: str = "data/clips"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    ffmpeg_log_level: str = "error"


@dataclass(frozen=True)
class LoadConfig:
    """Defaults for clip → RawNode batch loading.

    Attributes:
        annotations_file: M3-Bench annotations JSON.
        clips_root: Where sliced clips live.
        videos_root: Where the original videos live (used to resolve
            ``info["video_path"]`` in the manifest and for source
            metadata on the RawNode).
        output_jsonl: Where to write per-video sample records (M3-Agent
            compatible format).
        account_id: Tenant id; defaults to M3BENCH_ACCOUNT_ID.
        clip_seconds: Clip length in seconds — must match the value the
            slicer used, otherwise ``start_sec`` on every clip will be
            wrong. Defaults to the M3-Agent value (30).
        only_annotated: If True (default) only videos listed in the
            annotation JSON are processed; otherwise every ``*.mp4``
            under ``clips_root`` is processed.
    """
    annotations_file: str = "data/annotations/robot.json"
    clips_root: str = "data/clips"
    videos_root: str = "data/videos"
    output_jsonl: str = "data/data.jsonl"
    account_id: str = M3BENCH_ACCOUNT_ID
    clip_seconds: int = 30
    only_annotated: bool = True


def video_id_from_path(video_path: str) -> str:
    """Extract the M3 video id from a path like ``data/videos/robot/living_room_06.mp4``.

    Returns the filename stem. Works for both ``robot`` and ``web`` subsets.
    """
    return os.path.splitext(os.path.basename(video_path))[0]




# Default config for M3 scripts when no explicit --config is passed.  We use
# deploy/ogmemory.yaml in this repository so M3 experiments share the same
# model registry as the deployed service instead of loading ad-hoc .env files.
DEFAULT_OGMEM_CONFIG = str((Path(__file__).resolve().parent.parent / "deploy" / "ogmemory.yaml"))


@dataclass(frozen=True)
class M3ModelConfig:
    """Resolved model/config values used by M3 CLI scripts."""

    provider: str = "mock"
    api_key: str | None = None
    base_url: str | None = None
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096
    request_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class M3RuntimeConfig:
    """M3-specific config block from ogmemory.yaml."""

    episodic_llm: M3ModelConfig
    qa_llm: M3ModelConfig
    judge_llm: M3ModelConfig
    embedding_provider: str
    embedding_model: str
    embedding_base_url: str | None
    embedding_api_key: str | None
    embedding_dimension: int | None
    graph_connection_string: str | None
    # M3 vector-index config. The M3 pipeline writes embeddings into a
    # dedicated OpenGaussVectorIndex (default backend) so that
    # ``graph.search_by_embedding`` can route through HNSW + scope
    # columns instead of loading every node's JSONB into Python.
    # ``vector_index_connection_string`` is **not** a separate YAML
    # key — it inherits from the top-level ``vector_db.connection_string``
    # (same Postgres / openGauss instance as the graph store, different
    # table). Only ``backend`` / ``table_name`` / ``pool_size`` are
    # read from ``m3.vector_index``.
    vector_index_backend: str = "opengauss"
    vector_index_connection_string: str | None = None
    vector_index_table_name: str = "m3_episodic_vectors"
    vector_index_pool_size: int = 5
    embedding_batch_size: int = 2048

    def __post_init__(self):
        if (
            isinstance(self.embedding_batch_size, bool)
            or not isinstance(self.embedding_batch_size, int)
            or not 1 <= self.embedding_batch_size <= 2048
        ):
            raise ValueError("embedding_batch_size must be an integer between 1 and 2048")


def _load_yaml_config(config_path: str | None = None) -> dict[str, Any]:
    """Load ogmemory YAML without consulting .env files."""
    import yaml

    path = Path(config_path or os.environ.get("OGMEM_CONFIG") or DEFAULT_OGMEM_CONFIG)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _expand_env_ref(value: Any) -> Any:
    """Resolve simple ${VAR} strings against the current environment.

    deploy/ogmemory.yaml intentionally contains deploy-time placeholders. M3
    scripts do not load .env, but if the shell already exported the referenced
    variable we use it; otherwise the value is treated as unset.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    return value


def _pick(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    value = _expand_env_ref(mapping.get(key))
    return default if value in (None, "") else value


@dataclass(frozen=True)
class CVServiceConfig:
    """Connection to the bundled or an externally deployed compatible CV service."""

    base_url: str = "http://127.0.0.1:8081"
    timeout_seconds: float = 30.0
    auth_token: str | None = field(default=None, repr=False)


def load_cv_service_config(
    config_path: str | None = None,
    *,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    auth_token: str | None = None,
) -> CVServiceConfig:
    """Resolve explicit options > exported environment > YAML > Compose defaults.

    Like the other M3 loaders, this does not source deploy/.env. YAML values
    may reference exported environment variables using ${NAME}.
    """
    block = _load_yaml_config(config_path).get("cv_service")
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError("cv_service must be a mapping")
    url = base_url or os.environ.get("OGMEM_CV_SERVICE_URL") or _pick(
        block, "base_url", "http://127.0.0.1:8081",
    )
    if not isinstance(url, str):
        raise ValueError("cv_service.base_url must be an HTTP(S) service URL")
    try:
        parsed = urlsplit(url)
        valid_url = parsed.scheme in ("http", "https") and parsed.hostname and parsed.port != 0
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ValueError("cv_service.base_url must be an HTTP(S) service URL")

    timeout = timeout_seconds if timeout_seconds is not None else _pick(
        block, "timeout_seconds", 30.0,
    )
    try:
        if isinstance(timeout, bool):
            raise ValueError
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("cv_service.timeout_seconds must be a positive finite number") from None
    token = auth_token if auth_token is not None else (
        os.environ.get("OGMEM_CV_AUTH_TOKEN") or _pick(block, "auth_token")
    )
    if token is not None and not isinstance(token, str):
        raise ValueError("cv_service.auth_token must be a string")
    return CVServiceConfig(base_url=url.rstrip("/"), timeout_seconds=timeout, auth_token=token)


def _model_from_block(block: dict[str, Any], *, default_provider: str, default_model: str, default_temperature: float) -> M3ModelConfig:
    timeout = _pick(block, "request_timeout_seconds", 120.0)
    try:
        if isinstance(timeout, bool):
            raise ValueError
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            "M3 LLM request_timeout_seconds must be a positive finite number"
        ) from None
    return M3ModelConfig(
        provider=str(_pick(block, "provider", default_provider)),
        api_key=_pick(block, "api_key"),
        base_url=_pick(block, "base_url"),
        model=str(_pick(block, "model", default_model)),
        temperature=float(_pick(block, "temperature", default_temperature)),
        max_tokens=int(_pick(block, "max_tokens", 4096)),
        request_timeout_seconds=timeout,
    )


def load_m3_runtime_config(config_path: str | None = None) -> M3RuntimeConfig:
    """Resolve the M3 config from deploy/ogmemory.yaml.

    The optional top-level ``m3`` block can override the service defaults:

    ``m3.episodic_llm`` for multimodal clip summarization,
    ``m3.qa_llm`` for answer generation,
    ``m3.judge_llm`` for the eval judge,
    ``m3.embedding`` for episodic retrieval, and
    ``m3.vector_index`` for M3-specific vector index knobs
    (``table_name`` / ``backend`` / ``pool_size``).

    The M3 vector index **inherits its connection string from the
    top-level ``vector_db.connection_string``** — both the seed
    retriever and the M3 VectorIndex live on the same Postgres /
    openGauss instance and only differ in which table they use
    (``vector_index`` vs ``m3_episodic_vectors``). Declaring a
    separate ``m3.vector_index.connection_string`` was redundant
    configuration; if a deployment ever needs to point M3 at a
    dedicated cluster, override ``vector_db.connection_string``
    (the M3 block is now reserved for M3-specific knobs only).

    Missing values fall back to top-level ``llm`` / ``embedding`` /
    ``vector_db``.
    """
    raw = _load_yaml_config(config_path)
    llm = raw.get("llm") or {}
    emb = raw.get("embedding") or {}
    vdb = raw.get("vector_db") or {}
    storage = raw.get("storage") or {}
    m3 = raw.get("m3") or {}
    m3_ep = m3.get("episodic_llm") or {}
    m3_qa = m3.get("qa_llm") or {}
    m3_judge = m3.get("judge_llm") or {}
    m3_emb = m3.get("embedding") or {}
    m3_vi = m3.get("vector_index") or {}
    m3_graph = m3.get("graph_store") or {}

    episodic_base = {**llm, **m3_ep}
    qa_base = {**llm, **m3_qa}
    judge_base = {**llm, **m3_judge}
    embedding_base = {**emb, **m3_emb}
    # Vector-index resolution: the M3 vector index inherits its DSN
    # from the top-level ``vector_db.connection_string`` rather than
    # declaring its own. The graph store (graph_nodes / graph_edges)
    # and the vector index do **not** have to share a database —
    # in the recommended split topology the graph store sits on the
    # STORAGE_DB Postgres alongside the rest of the SQL state
    # (context_files / events / grants), while the vector index sits
    # on the VECTOR_DB (openGauss in production, or a single-DB
    # pgvector Postgres when ``m3.vector_index.backend=pgvector``).
    #
    # Graph-store DSN resolution order:
    #   1. ``m3.graph_store.connection_string``  (explicit override)
    #   2. ``storage.connection_string``         (default; same DB
    #                                             as ContextFS / etc.)
    # ``vector_db.connection_string`` is intentionally **not** in the
    # fallback chain — the graph store and the vector index are
    # separate roles and merging them is what glued M3 to the vector
    # DB through the legacy single-DB topology. Callers that want the
    # legacy behaviour (graph store + vector index on the same
    # openGauss cluster) can set ``m3.graph_store.connection_string``
    # to the same DSN as ``vector_db.connection_string``.
    vi_connection = _pick(vdb, "connection_string")
    vi_table = _pick(m3_vi, "table_name") or "m3_episodic_vectors"
    vi_pool_size = _pick(m3_vi, "pool_size") or 5
    vi_backend = _pick(m3_vi, "backend") or "opengauss"

    # Graph-store DSN: m3 override wins, otherwise follow storage.
    graph_connection_string = (
        _pick(m3_graph, "connection_string")
        or _pick(storage, "connection_string")
    )

    return M3RuntimeConfig(
        episodic_llm=_model_from_block(
            episodic_base,
            default_provider="minimax",
            default_model="MiniMax-M3",
            default_temperature=0.2,
        ),
        qa_llm=_model_from_block(
            qa_base,
            default_provider="openai",
            default_model="gpt-4o-mini",
            default_temperature=0.2,
        ),
        # The judge defaults to MiniMax-M3 — that's the model the
        # bench-scale eval harness was built around and the only one
        # guaranteed to have OGMEM_MINIMAX_API_KEY in deploy.env. Users
        # who want a different judge (DeepSeek, vLLM, OpenAI) override
        # ``m3.judge_llm`` in ogmemory.yaml or pass --judge-base-url /
        # --judge-model / --judge-api-key on the CLI.
        judge_llm=_model_from_block(
            judge_base,
            default_provider="minimax",
            default_model="MiniMax-M3",
            default_temperature=0.0,
        ),
        embedding_provider=str(_pick(embedding_base, "provider", "openai")),
        embedding_model=str(_pick(embedding_base, "model", "text-embedding-ada-002")),
        embedding_base_url=_pick(embedding_base, "base_url"),
        embedding_api_key=_pick(embedding_base, "api_key"),
        embedding_batch_size=_pick(embedding_base, "batch_size", 2048),
        embedding_dimension=(
            int(_pick(embedding_base, "dimension"))
            if _pick(embedding_base, "dimension") is not None
            else (
                int(_pick(vdb, "dimension"))
                if _pick(vdb, "dimension") is not None
                else None
            )
        ),
        graph_connection_string=graph_connection_string,
        vector_index_backend=str(vi_backend),
        vector_index_connection_string=vi_connection,
        vector_index_table_name=str(vi_table),
        vector_index_pool_size=int(vi_pool_size),
    )
def resolve_video_path(
    declared: str,
    *,
    project_root: Path,
    videos_root: Path,
) -> Path | None:
    """Resolve a video path declared in the M3 annotation JSON to a real file.

    Tries, in order:
    1. ``<project_root> / declared`` — the canonical M3 layout where the
       script is invoked from the project root (data/videos/... lives
       under cwd).
    2. ``<videos_root> / <subset> / <stem>.mp4`` — keeps the
       ``videos_root`` argument useful as an override for relocated data
       (e.g. downloaded to a different drive).
    3. ``<videos_root> / <stem>.mp4`` — last-resort flat fallback for
       annotations that don't use a subset directory.

    Returns the first existing candidate, or ``None`` if none match.
    """
    stem = video_id_from_path(declared)
    parts = Path(declared).parts
    subset: str | None = None
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "videos":
        subset = parts[2] if len(parts) >= 3 else None

    candidates: list[Path | None] = [
        project_root / declared,
        videos_root / subset / f"{stem}.mp4" if subset else None,
        videos_root / f"{stem}.mp4",
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c
    return None
