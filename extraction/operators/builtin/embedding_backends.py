"""Speaker embedding backend implementations.

Wraps an ECAPA-TDNN model (default ``speechbrain/spkrec-ecapa-voxceleb``)
into a 192-dim speaker embedding per audio slice. Two impls:

* :class:`SpeechBrainECAPABackend` — uses ``speechbrain.inference
  .speaker.EncoderClassifier``. CPU-runnable. Default.

* :class:`MockSpeakerEmbeddingBackend` — deterministic stub. Returns
  a sha1-derived 192-dim vector based on (audio path, start_sec,
  end_sec, speaker_id) so unit tests can pin results.

Heavy deps (``speechbrain``, ``torch``, ``torchaudio``) are lazy-
imported so ``import extraction.operators.builtin`` works without
them.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class SpeakerEmbedding:
    """One speaker's 192-dim embedding vector + provenance."""

    speaker_id: str
    embedding: list[float]
    start_sec: float
    end_sec: float
    model: str


@dataclass
class SpeakerEmbeddingResult:
    """Top-level output of a speaker-embedding backend."""

    embeddings: list[SpeakerEmbedding] = field(default_factory=list)
    model: str = ""


class SpeakerEmbeddingBackendError(Exception):
    """Raised by a speaker-embedding backend on hard failure."""

    def __init__(self, backend: str, message: str) -> None:
        super().__init__(f"[{backend}] {message}")
        self.backend = backend


class SpeakerEmbeddingTooShortError(SpeakerEmbeddingBackendError):
    """Valid audio, but insufficient samples for a speaker vector."""

    def __init__(self, backend: str, duration_sec: float) -> None:
        super().__init__(backend, f"audio too short for embedding: {duration_sec:.6f}s")
        self.duration_sec = duration_sec


class SpeakerEmbeddingBackend(Protocol):

    """Backend interface for :class:`SpeakerEmbeddingExtractor`."""

    @property
    def name(self) -> str: ...

    @property
    def embedding_dim(self) -> int: ...

    def embed(
        self,
        wav_path: str,
        *,
        start_sec: float,
        end_sec: float,
        speaker_id: str = "",
    ) -> list[float]: ...


# ---------------------------------------------------------------------------
# SpeechBrain ECAPA-TDNN backend
# ---------------------------------------------------------------------------

def _nearest_existing_ancestor_writable(path: str) -> bool:
    """True when ``path`` can be created: its nearest existing ancestor is
    writable. Walks up until an existing directory is found — no side effects.
    """
    probe = path
    while True:
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        if os.path.isdir(probe):
            return os.access(probe, os.W_OK)
        probe = parent
    return os.path.isdir(probe) and os.access(probe, os.W_OK)


class SpeechBrainECAPABackend:
    """speechbrain/spkrec-ecapa-voxceleb (192-dim ECAPA-TDNN).

    Loads via ``EncoderClassifier.from_hparams``. The model is
    downloaded on first use (~80 MB) and cached under
    ``~/.cache/speechbrain/<model_name>/`` (the ``savedir`` we pass).

    Construction is lazy — the model isn't loaded until the first
    ``embed`` call. That keeps imports cheap.
    """

    DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
    DEFAULT_DIM = 192

    def __init__(
        self,
        model: str | None = None,
        *,
        device: str = "cpu",
        savedir: str | None = None,
    ) -> None:
        self._model_name = model or self.DEFAULT_MODEL
        self._device = device
        self._savedir = savedir or self._default_savedir()
        self._classifier: Any = None

    def _default_savedir(self) -> str:
        """Resolve a writable speechbrain model cache directory.

        The legacy default ``~/.cache/speechbrain/<model>`` breaks in
        containers that ship ``HOME=/`` (the m3-runner image does): it
        resolves to ``/.cache/...``, which is not writable, and every
        ECAPA embed then fails with ``[Errno 13] Permission denied``
        (observed: 71/79 clips failed before an explicit ``HOME`` was
        set). Probe the legacy path first, then ``XDG_CACHE_HOME``, then
        ``HF_HOME`` (the m3 runners mount a persistent HF cache there),
        then the tempdir — first candidate whose nearest existing
        ancestor is writable wins. Nothing is created here; probing is
        side-effect free so ``__init__`` stays cheap and test-friendly.
        """
        dirname = self._model_name.replace("/", "_")
        home = os.path.expanduser("~")
        candidates: list[str] = []
        if home and home != os.path.sep:
            candidates.append(os.path.join(home, ".cache", "speechbrain", dirname))
        for env in ("XDG_CACHE_HOME", "HF_HOME"):
            base = os.environ.get(env)
            if base:
                candidates.append(os.path.join(base, "speechbrain", dirname))
        candidates.append(os.path.join(tempfile.gettempdir(), "speechbrain", dirname))
        for candidate in candidates:
            if _nearest_existing_ancestor_writable(candidate):
                return candidate
        # Nothing probed writable (unusual) — keep the legacy path so the
        # speechbrain load-time error names a familiar location.
        return os.path.join(home, ".cache", "speechbrain", dirname)

    @property
    def name(self) -> str:
        return "speechbrain"

    @property
    def embedding_dim(self) -> int:
        return self.DEFAULT_DIM

    def _load(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:  # pragma: no cover
            raise SpeakerEmbeddingBackendError(
                self.name,
                "speechbrain / torch not installed. "
                "Install with `uv pip install -e .[asr]` "
                "or use --backend mock.",
            ) from exc
        try:
            self._classifier = EncoderClassifier.from_hparams(
                source=self._model_name,
                savedir=self._savedir,
                run_opts={"device": self._device},
            )
        except Exception as exc:  # noqa: BLE001
            raise SpeakerEmbeddingBackendError(
                self.name,
                f"failed to load {self._model_name}: {exc}",
            ) from exc
        return self._classifier

    def embed(
        self,
        wav_path: str,
        *,
        start_sec: float,
        end_sec: float,
        speaker_id: str = "",
    ) -> list[float]:
        try:
            import torch
            import torchaudio
        except ImportError as exc:  # pragma: no cover
            raise SpeakerEmbeddingBackendError(
                self.name,
                "torch / torchaudio not installed.",
            ) from exc
        try:
            wav, sr = torchaudio.load(wav_path)
        except Exception as exc:  # noqa: BLE001
            raise SpeakerEmbeddingBackendError(
                self.name, f"torchaudio.load({wav_path}) failed: {exc}",
            ) from exc
        if sr != 16000:
            try:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            except Exception as exc:  # noqa: BLE001
                raise SpeakerEmbeddingBackendError(
                    self.name,
                    f"resample({sr} -> 16000) failed: {exc}",
                ) from exc
            sr = 16000
        # Mono
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        s = max(0, int(round(start_sec * sr)))
        e = min(wav.shape[-1], int(round(end_sec * sr)))
        if e <= s:
            raise SpeakerEmbeddingBackendError(self.name, "empty or out-of-bounds audio range")
        if e - s < int(0.1 * sr):   # < 100 ms — too short
            raise SpeakerEmbeddingTooShortError(self.name, (e - s) / sr)
        signal = wav[:, s:e]
        classifier = self._load()
        try:
            with torch.no_grad():
                emb = classifier.encode_batch(signal)
            emb = emb.squeeze().detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            raise SpeakerEmbeddingBackendError(
                self.name,
                f"encode_batch failed: {exc}",
            ) from exc
        # L2-normalize so cosine distance matches dot product on
        # the unit sphere (matches ECAPA reference impl).
        norm = float(math.sqrt(sum(float(x) * float(x) for x in emb)) or 1.0)
        return [float(x) / norm for x in emb]


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockSpeakerEmbeddingBackend:
    """Deterministic fake speaker embeddings.

    Generates a 192-dim unit vector derived from a sha1 of the
    (audio path, start_sec, end_sec, speaker_id) tuple. Same input
    always returns the same vector — unit tests can pin it.
    """

    DEFAULT_DIM = 192

    def __init__(
        self,
        *,
        embedding_dim: int = DEFAULT_DIM,
        fail_on_path_substring: str | None = None,
    ) -> None:
        self._dim = int(embedding_dim)
        self._fail_on_path_substring = fail_on_path_substring

    @property
    def name(self) -> str:
        return "mock"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(
        self,
        wav_path: str,
        *,
        start_sec: float,
        end_sec: float,
        speaker_id: str = "",
    ) -> list[float]:
        if self._fail_on_path_substring and self._fail_on_path_substring in wav_path:
            raise SpeakerEmbeddingBackendError(self.name, f"forced failure on {wav_path}")
        seed = f"{wav_path}|{start_sec:.4f}|{end_sec:.4f}|{speaker_id}".encode()
        digest = hashlib.sha512(seed).digest()
        # Tile the digest until we have >= dim floats, then take
        # the first dim. Each byte pair becomes one float in [-1, 1].
        vec: list[float] = []
        i = 0
        while len(vec) < self._dim:
            pair = digest[(i % (len(digest) - 1))] + 256 * digest[(i + 1) % len(digest)]
            vec.append((pair / 65535.0) * 2.0 - 1.0)
            i += 2
        vec = vec[: self._dim]
        # L2-normalise so cosine distance is meaningful.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_speaker_embedding_backend(
    backend: str,
    *,
    model: str | None = None,
    device: str = "cpu",
    embedding_dim: int = 192,
) -> SpeakerEmbeddingBackend:
    """Construct a speaker-embedding backend from CLI kwargs."""
    backend = (backend or "speechbrain").lower()
    if backend in ("speechbrain", "ecapa", "ecapa-tdnn"):
        return SpeechBrainECAPABackend(model=model, device=device)
    if backend in ("mock",):
        return MockSpeakerEmbeddingBackend(embedding_dim=embedding_dim)
    raise SpeakerEmbeddingBackendError(
        backend,
        f"unknown speaker-embedding backend: {backend!r}. "
        "Expected one of: speechbrain, mock.",
    )
