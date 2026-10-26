"""Shared deterministic neighborhood planning, separate from query ranking."""
from heapq import merge

from core.clip_lookup import clip_ordinal


def validate_context_options(radius: int, context_k: int) -> None:
    for name, value in (("include_neighbors", radius), ("context_k", context_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0")
    if context_k and not radius:
        raise ValueError("context_k > 0 requires include_neighbors > 0")


def neighbor_plan(seeds, *, radius, before_clip=None, max_clips=None):
    """Map (video, clip) to (distance, seed rank, supporting seed identity).

    Nearest clips first, then stronger main evidence; no invented clip IDs for
    malformed provenance. Existing main clips are left out of context requests.
    Merge directional streams so sparse/invalid ranges do not require walking
    the entire radius before reaching the first eligible clip.
    """
    if max_clips is not None and max_clips <= 0:
        return {}
    normalized = [(video, clip_ordinal(clip), identity) for video, clip, identity in seeds]
    occupied = {(video, clip) for video, clip, _ in normalized if clip is not None}

    def direction(video, clip, identity, rank, step):
        if step == -1:
            start = max(1, clip - before_clip) if before_clip is not None else 1
            end = min(radius, clip)
        else:
            start = 1
            end = min(radius, before_clip - clip) if before_clip is not None else radius
        for distance in range(start, end + 1):
            target = clip + step * distance
            if (video, target) not in occupied:
                yield (distance, rank, video, target), identity

    streams = [
        direction(video, clip, identity, rank, step)
        for rank, (video, clip, identity) in enumerate(normalized)
        if video and clip is not None
        for step in (-1, 1)
    ]
    planned = {}
    for (distance, rank, video, target), identity in merge(*streams):
        key = (video, target)
        if key not in planned:
            planned[key] = (distance, rank, identity)
            if max_clips is not None and len(planned) >= max_clips:
                break
    return planned
