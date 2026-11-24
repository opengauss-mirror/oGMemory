"""Query-time fusion of the evidence time channel."""

import math
import re
from datetime import UTC, datetime

from core.evidence import resolve_event_time
from core.operation_deadline import current_deadline, deadline_expired


def date_prefixes(query):
    prefixes = re.findall(r"\b\d{4}-\d{2}(?:-\d{2})?\b", query)
    prefixes.extend(f"{year}-{int(month):02d}" for year, month in re.findall(r"(\d{4})年(\d{1,2})月", query))
    months = "January February March April May June July August September October November December".split()
    for month, year in re.findall(r"\b([A-Za-z]+)\s+(\d{4})\b", query):
        index = next((i for i, name in enumerate(months, 1) if month.casefold() in {name.casefold(), name[:3].casefold()}), None)
        if index:
            prefixes.append(f"{year}-{index:02d}")
    for year in re.findall(r"(?<!\d)(\d{4})(?!\d)", query):
        if not any(prefix.startswith(year) for prefix in prefixes):
            prefixes.append(year)
    return list(dict.fromkeys(prefixes))


def fuse_auxiliary(request, scores, units, sources):
    """Return fused scores, channel diagnostics and whether the deadline prevented fusion."""
    if deadline_expired(current_deadline()):
        return scores, {"route": "none", "rho": 1.0, "reason": "deadline"}, True
    dates = date_prefixes(request.query)
    todo = bool(re.search(r"\b(todo|pending|unfinished)\b|待办|未完成", request.query, re.I))
    auxiliary, rho, route = {}, 1.0, "none"
    if dates or todo:
        route = "time"
        if request.use_time:
            now = datetime.now(UTC)
            for key, unit in units.items():
                if deadline_expired(current_deadline()):
                    break
                source = sources[unit["source_key"]]
                event_time, _ = resolve_event_time(unit["text"], source["recorded_at"], source.get("timezone"))
                value = event_time or unit["recorded_at"]
                if not value:
                    continue
                if dates and any(value.startswith(prefix) for prefix in dates):
                    auxiliary[key] = 1.0
                elif todo and key in scores:
                    recorded = datetime.fromisoformat(value)
                    if recorded.tzinfo:
                        days = max(0, (now - recorded.astimezone(UTC)).total_seconds() / 86400)
                        auxiliary[key] = math.exp(-days / 30)
            if deadline_expired(current_deadline()):
                return scores, {"route": route, "rho": 1.0, "reason": "deadline"}, True
            auxiliary = dict(sorted(auxiliary.items(), key=lambda item: (-item[1], item[0]))[:5])
            rho = .7
    if not auxiliary:
        return scores, {"route": route, "rho": 1.0, "reason": "disabled_or_no_results"}, False
    content_max, auxiliary_max = max(scores.values(), default=0), max(auxiliary.values())
    fused = {key: rho * scores.get(key, 0) / (content_max or 1)
                  + (1 - rho) * auxiliary.get(key, 0) / auxiliary_max for key in set(scores) | set(auxiliary)}
    return fused, {"route": route, "rho": rho, "reason": "ready"}, False
