"""Sanitized diagnostics for persisted migration jobs."""
import re
import traceback


def _failure_diagnostic(error, cfg):
    from skills.sanitizer import sanitize_trace_text

    secrets = set()
    for name, value in vars(cfg).items():
        if isinstance(value, str) and re.search(
            r"key|password|secret|token|connection_string", name, re.I
        ):
            secrets.add(value)
            if "connection_string" in name and value:
                try:
                    from psycopg2.extensions import parse_dsn

                    secrets.add(parse_dsn(value).get("password", ""))
                except Exception:
                    pass  # A malformed DSN must not prevent failure reporting.
    secrets.discard("")
    parts, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        # Keep call locations, but never source lines or local variable values.
        frames = [
            f"  {frame.f_globals.get('__name__', '<unknown>')}.{frame.f_code.co_name}:{line}"
            for frame, line in traceback.walk_tb(error.__traceback__)
        ]
        reason = sanitize_trace_text(str(error))
        for secret in sorted(secrets, key=len, reverse=True):
            reason = reason.replace(secret, "[REDACTED_SECRET]")
        parts.append("\n".join([*frames, f"{type(error).__name__}: {reason}"]))
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return "\nException chain:\n".join(parts)
