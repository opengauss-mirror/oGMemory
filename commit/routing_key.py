"""Routing key normalization for consistent memory addressing.

LLM-generated routing keys are inconsistent across sessions.
This module normalizes them using a synonym table and simple heuristics
to prevent memory fragmentation (e.g., "coffee" vs "beverage" → same node).
"""

from core.memory_names import (
    canonical_memory_key as canonical_memory_key,
    is_meaningful_memory_key as is_meaningful_memory_key,
)

# Synonym groups: all keys in a group map to the first (canonical) entry.
_SYNONYM_GROUPS: list[list[str]] = [
    # Beverage
    ["beverage", "coffee", "tea", "drink"],
    # Commute
    ["commute", "transport", "transportation", "travel_to_work"],
    # Notes
    ["note_taking", "notes", "note_tool"],
    # Code
    ["coding_style", "code_style", "programming_style", "coding_preference"],
    # Music
    ["music", "music_taste", "music_preference"],
    # Food
    ["food", "food_preference", "diet", "eating"],
    # Exercise
    ["exercise", "workout", "fitness", "gym"],
    # Reading
    ["reading", "books", "reading_preference"],
    # Sleep
    ["sleep", "sleep_habit", "sleep_schedule"],
    # Weather
    ["weather", "weather_preference"],
    # Pets
    ["pets", "pet", "animals"],
]

# Build reverse lookup: any synonym → canonical key
_SYNONYM_MAP: dict[str, str] = {}
for group in _SYNONYM_GROUPS:
    canonical = group[0]
    for synonym in group:
        _SYNONYM_MAP[synonym] = canonical


def normalize_routing_key(raw_key: str, category: str) -> str:
    """Normalize a routing key for consistent addressing.

    Steps:
    1. Lowercase + strip whitespace
    2. Replace spaces/hyphens with underscores
    3. Lookup synonym table
    4. Return normalized key

    Args:
        raw_key: LLM-generated routing key
        category: Candidate category (preference, entity, pattern, etc.)

    Returns:
        Normalized routing key
    """
    if not raw_key:
        return category

    # Step 1-2: Basic normalization
    key = raw_key.lower().strip().replace(" ", "_").replace("-", "_")

    # Remove consecutive underscores
    while "__" in key:
        key = key.replace("__", "_")

    # Step 3: Synonym lookup
    if key in _SYNONYM_MAP:
        return _SYNONYM_MAP[key]

    return key
