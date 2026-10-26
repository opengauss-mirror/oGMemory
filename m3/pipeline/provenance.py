"""Fail closed when a work directory belongs to a different media input.

Graph clip IDs are stable per video_id/clip_id: replacing only local files
cannot safely invalidate their derived graph/vector records. Changed inputs
must use a new work directory AND video_id (or an explicitly cleaned graph).
"""
import hashlib
import json
from pathlib import Path


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def validate_input(ctx, spec):
    from m3.pipeline.runner import _format_value
    relevant = {}
    for stage in spec.stages:
        # Execution switches are not media identity. A targeted resume disables
        # completed stages but must still validate their original parameters;
        # dropping them would reject an unchanged input (or hide a changed
        # parameter on a disabled stage).
        if stage.type in ("video.trim", "m3.preprocess", "m3.load_clips"):
            relevant[stage.id] = _format_value(stage.params, ctx.variables)
    identity = {"source": file_identity(ctx.video), "seconds": ctx.seconds,
                "video_id": ctx.video_id, "account_id": ctx.account_id,
                "stages": relevant}
    path = ctx.work_dir / "input_provenance.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise RuntimeError("Media input/configuration changed: use a new --work-dir AND "
                               "--video-id to avoid reusing stale graph/vector evidence.")
    else:
        selected = ctx.artifact("selected_video", optional=True)
        clips = ctx.artifact("clips_root", optional=True)
        if ((selected and selected.exists() and selected.resolve() != ctx.video.resolve())
                or (clips and clips.exists() and any(clips.rglob("*.mp4")))):
            raise RuntimeError("Existing media artifacts have no input provenance. Use a new "
                               "--work-dir AND --video-id; legacy cache cannot be verified.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")


def stage_identity(ctx, stage):
    source = ctx.video if stage.type == "video.trim" else ctx.artifact("selected_video")
    return {"source": file_identity(source), "params": stage.params,
            "seconds": ctx.seconds, "video_id": ctx.video_id}


def manifest_path(ctx, stage):
    return ctx.work_dir / f".{stage.id}.provenance.json"


def outputs(ctx, stage):
    if stage.type == "video.trim":
        return [ctx.artifact("selected_video")]
    return sorted(ctx.artifact("clips_root").rglob("*.mp4"))


def stage_complete(ctx, stage):
    path = manifest_path(ctx, stage)
    if not path.exists():
        return False
    try:
        saved = json.loads(path.read_text())
        files = outputs(ctx, stage)
        return (bool(files) and saved["input"] == stage_identity(ctx, stage)
                and saved["outputs"] == [file_identity(p) for p in files]
                and all(p.stat().st_size > 0 for p in files))
    except (OSError, ValueError, KeyError):
        return False


def record_stage(ctx, stage):
    manifest_path(ctx, stage).write_text(json.dumps({
        "input": stage_identity(ctx, stage),
        "outputs": [file_identity(p) for p in outputs(ctx, stage)],
    }, sort_keys=True), encoding="utf-8")
