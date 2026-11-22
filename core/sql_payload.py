"""Text representation constraints shared by SQL content and index writers."""


def validate_sql_strings(value):
    """Reject actual NUL in text or JSON keys/values, preserving literal escapes."""
    pending = [value]
    while pending:
        value = pending.pop()
        if isinstance(value, str) and "\x00" in value:
            raise ValueError("SQL target cannot store NUL characters")
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
