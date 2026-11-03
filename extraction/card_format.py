"""Check the Markdown card contract before accepting generated memories."""

from core.errors import MemoryCardFormatError


def validate_memory_card(memory):
    if memory.category not in {"entity", "preference"}:
        return
    errors = []
    if len((memory.abstract or "").strip().splitlines()) != 1:
        errors.append("abstract must be one nonempty line")
    overview = [line.strip() for line in (memory.overview or "").splitlines() if line.strip()]
    if not overview or any(not line.startswith("- ") for line in overview):
        errors.append("overview must contain one core fact per '- ' bullet")
    lines = [line.rstrip() for line in (memory.content or "").splitlines() if line.strip()]
    if not lines or not lines[0].startswith("# ") or not lines[0][2:].strip():
        errors.append("content must begin with one H1 display name")
    section = None
    bullets = 0
    sections = 0
    intro = 0
    for line in lines[1:]:
        if line.startswith("## ") and memory.category == "entity":
            if section is not None and not bullets:
                errors.append("entity sections must contain fact bullets")
            section, bullets = line[3:].strip(), 0
            sections += 1
            if not section:
                errors.append("entity section names must be nonempty")
        elif line.startswith("- ") and line[2:].strip():
            if memory.category == "entity" and section is None:
                errors.append("entity facts must be under an H2 section")
            bullets += 1
        elif line.startswith("  ") and bullets:
            continue
        elif memory.category == "entity" and section is None and intro == 0 and not line.startswith("#"):
            intro += 1
        else:
            errors.append("content must use fact bullets instead of paragraph bodies")
    if not bullets or memory.category == "entity" and not sections:
        errors.append("content must contain fact bullets and entity cards need H2 sections")
    if errors:
        raise MemoryCardFormatError(
            f"{memory.category}/{memory.routing_key}: " + "; ".join(dict.fromkeys(errors)))
