"""Build a credential-free release ZIP from an explicit file allowlist."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "plugin.json", "mcp.json", "config.example.json", "README.md", "README_zh.md",
    "servers/config.mjs", "servers/mcp-proxy.mjs", "scripts/doctor.mjs",
    "skills/ogmemory-memory/SKILL.md",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    version = json.loads((ROOT / "plugin.json").read_text())["version"]
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"ogmemory-agent-plugin-{version}.zip"
    entries = []
    for name in FILES:
        content = (ROOT / name).read_bytes()
        if name.endswith(".md"):
            content = content.decode().replace("../../docs/integrations/agent-plugins.md", "CLIENT-GUIDE.md").encode()
        entries.append((f"agent-plugins/{name}", content))
    entries.append(("agent-plugins/LICENSE", (ROOT.parents[1] / "LICENSE").read_bytes()))
    entries.append(("agent-plugins/ACCEPTANCE.json", (ROOT.parents[1] / "docs/integrations/agent-plugins-acceptance.json").read_bytes()))
    for source, destination in [("agent-plugins.md", "CLIENT-GUIDE.md")]:
        text = (ROOT.parents[1] / "docs/integrations" / source).read_text(encoding="utf-8")
        def link(match):
            target = match[1]
            if "://" in target or target.startswith("#"):
                return match[0]
            if target == "agent-plugins.md":
                return "](CLIENT-GUIDE.md)"
            if target == "agent-plugins-acceptance.json":
                return "](ACCEPTANCE.json)"
            path = (ROOT.parents[1] / "docs/integrations" / target).resolve().relative_to(ROOT.parents[1])
            return "](https://gitcode.com/opengauss/oGMemory/blob/dev/" + path.as_posix() + ")"
        entries.append((f"agent-plugins/{destination}", re.sub(r"\]\(([^)]+)\)", link, text).encode()))
    with ZipFile(archive, "x", ZIP_DEFLATED) as target:
        for name, content in entries:
            target.writestr(name, content)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(archive.resolve())


if __name__ == "__main__":
    main()
