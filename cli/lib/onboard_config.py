"""Preserve settings outside the wizard and save configuration atomically."""
import time
from copy import deepcopy
from pathlib import Path

import yaml

from cli.lib.local_runtime import private_write


def read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError()
        for key in ("llm", "embedding", "vector_db", "storage", "service", "paths", "identity"):
            if key in value and not isinstance(value[key], dict):
                raise ValueError()
        return value
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError("Existing configuration is not a valid YAML mapping; it was not changed") from exc


def merge_config(existing: dict, generated: dict) -> dict:
    result = deepcopy(existing)
    owned = {"llm": {"provider", "api_key", "base_url", "model"},
             "embedding": {"provider", "api_key", "base_url", "model"},
             "vector_db": {"type", "dimension", "connection_string", "milvus_uri",
                           "milvus_collection", "milvus_db_name", "milvus_timeout"},
             "storage": {"backend", "dialect", "connection_string"},
             "service": {"http_port"}, "plugin": {"type", "openclaw_config"}}
    for section, values in generated.items():
        current = result.setdefault(section, {})
        for key, value in values.items():
            if key not in current or key in owned.get(section, set()):
                current[key] = value
    return result


def save_config(path: Path, value: dict) -> bool:
    if path.exists() and read_config(path) == value:
        return False
    if path.exists():
        backup = path.with_name(path.name + f".bak-{time.time_ns()}")
        private_write(backup, path.read_text(encoding="utf-8"))
        print(f"Configuration backup: {backup}")
    private_write(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
    return True


def changed_fields(old: dict, new: dict, prefix="") -> list[str]:
    changed = []
    for key in sorted(old.keys() | new.keys()):
        name = f"{prefix}{key}"
        if isinstance(old.get(key), dict) and isinstance(new.get(key), dict):
            changed.extend(changed_fields(old[key], new[key], name + "."))
        elif old.get(key) != new.get(key):
            changed.append(name)
    return changed


def existing_defaults(args, raw: dict) -> None:
    fields = {"provider": ("llm", "provider"), "model": ("llm", "model"),
              "api_key": ("llm", "api_key"), "base_url": ("llm", "base_url"),
              "embedding_provider": ("embedding", "provider"), "embedding_model": ("embedding", "model"),
              "embedding_api_key": ("embedding", "api_key"), "embedding_base_url": ("embedding", "base_url"),
              "embedding_dimension": ("vector_db", "dimension"), "vector_db": ("vector_db", "type"),
              "vector_db_connection": ("vector_db", "connection_string"),
              "milvus_uri": ("vector_db", "milvus_uri"),
              "milvus_collection": ("vector_db", "milvus_collection"),
              "milvus_db_name": ("vector_db", "milvus_db_name"),
              "milvus_timeout": ("vector_db", "milvus_timeout"),
              "db_connection": ("storage", "connection_string"), "sql_dialect": ("storage", "dialect"),
              "storage_backend": ("storage", "backend"), "port": ("service", "http_port")}
    old_embedding = raw.get("embedding", {})
    old_provider = old_embedding.get("provider")
    old_model = old_embedding.get("model")
    provider_changed = old_provider is not None and getattr(
        args, "embedding_provider", None,
    ) not in (None, old_provider)
    model_changed = old_model is not None and getattr(
        args, "embedding_model", None,
    ) not in (None, old_model)
    for name, (section, key) in fields.items():
        provider_arg = "provider" if section == "llm" else "embedding_provider"
        if section in {"llm", "embedding"} and getattr(args, provider_arg, None) not in (None, raw.get(section, {}).get("provider")):
            continue
        if name == "embedding_dimension" and (provider_changed or model_changed):
            continue
        if (getattr(args, name, None) is None or (name in {"port", "storage_backend", "sql_dialect"} and name not in getattr(args, "_explicit_fields", set()))) and key in raw.get(section, {}):
            setattr(args, name, raw[section][key])
    if getattr(args, "non_interactive", False) and model_changed and getattr(args, "embedding_dimension", None) is None:
        raise ValueError("--embedding-dimension is required when changing the embedding model")
    for name, default in (("port", 8090), ("storage_backend", "sql"), ("sql_dialect", "opengauss")):
        if getattr(args, name, None) is None:
            setattr(args, name, default)
