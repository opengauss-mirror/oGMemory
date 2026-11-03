"""Passive BEAM instrumentation, loaded ONLY via an explicit evaluation PYTHONPATH.

Mount this directory into the isolated memory container and add that directory
to PYTHONPATH, then set BEAM_COMPOSE_CAPTURE=1. The production image and source
are not changed. No prompts, routing, sampling, limits or return values change.
Never mount the evaluator, dataset, answer files or snapshots into the agent.
"""
from functools import wraps
import json
import os
from pathlib import Path
import uuid


def capture_first_compose(params, result, folder=Path("/tmp/ogmem-compose")):
    identity = params.get("sessionId")
    try:
        if not isinstance(identity, str) or str(uuid.UUID(identity)) != identity.lower():
            return
    except ValueError:
        return
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    folder = Path(folder)
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(folder / (identity + ".json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return  # First actual response is immutable, even on another compose.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def install():
    # Match server.app's cold-process import order. Importing memory_service
    # first enters commit -> core -> service.api while commit is incomplete.
    # This is bootstrap only; it does not construct services or call providers.
    import core  # noqa: F401
    from server.memory_service import MemoryService

    original = MemoryService.compose
    if getattr(original, "_beam_passive_capture", False):
        return

    @wraps(original)
    def compose(self, params):
        result = original(self, params)
        capture_first_compose(params, result)
        return result

    compose._beam_passive_capture = True
    MemoryService.compose = compose


if os.environ.get("BEAM_COMPOSE_CAPTURE") == "1":
    install()
