"""Allowlisted, informational environment descriptions; never configuration input."""

import os

from service.migration.package import PackageError

FIELDS = {
    "embedding": {"provider", "model", "dimension", "model_version"},
    "content_storage": {"backend", "database_type", "database_version"},
    "vector_storage": {"backend"},
}


def model_dimension(cfg, provider):
    if provider != "st":
        return getattr(cfg, "opengauss_dimension", None)
    # ST infers its dimension from the model, not the vector-store setting.
    # This is optional source metadata: unavailable models must not block export.
    embedder = None
    try:
        embedder = cfg.to_provider_config().create_embedder()
        dimension = embedder.dimension
        return dimension if type(dimension) is int and dimension > 0 else None
    except Exception:
        return None
    finally:
        close = getattr(embedder, "close", None)
        if callable(close):
            close()


def describe(cfg, adapter, *, load_model=True):
    provider = getattr(cfg, "embedding_provider", None) or getattr(cfg, "provider", None)
    vector_free = getattr(cfg, "agfs_retrieval_enabled", False)
    if vector_free:
        provider = "none"
    model = None
    if provider == "mock":
        model = "MockEmbedder"
    elif provider == "st":
        model = cfg.st_model
    elif provider in {"openai", "openai-cached"}:
        model = cfg.openai_embedding_model
    elif provider == "volcengine":
        model = os.environ.get(
            "VOLC_EMBEDDING_MODEL", cfg.openai_embedding_model or "doubao-embedding-vision-250615"
        )
    result = {
        "embedding": {
            "provider": provider,
            "model": model,
            "dimension": None if vector_free or (provider == "st" and not load_model) else model_dimension(cfg, provider),
            # Current providers do not expose a reliable resolved model revision.
            "model_version": None,
        },
        "content_storage": adapter.description(),
        "vector_storage": {"backend": "none" if vector_free else cfg.vector_db_type},
    }
    validate_description(result, adapter.backend)
    return result


def validate_description(value, backend):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise PackageError("invalid source environment description")
    for section, fields in FIELDS.items():
        data = value[section]
        if not isinstance(data, dict) or set(data) != fields:
            raise PackageError("unsupported source environment fields")
        for key, item in data.items():
            if item is None:
                continue
            if key == "dimension":
                if type(item) is not int or item <= 0:
                    raise PackageError("invalid source embedding dimension")
            elif not isinstance(item, str) or not item or len(item) > 512:
                raise PackageError("invalid source environment value")
    if value["content_storage"]["backend"] != backend:
        raise PackageError("source storage descriptions disagree")


def compatibility(source, target):
    changes = []
    if source is not None:
        for section, fields in FIELDS.items():
            for field in sorted(fields):
                before, after = source[section][field], target[section][field]
                if before != after:
                    changes.append(f"{section}.{field}")
    vector_free = target["embedding"]["provider"] == "none"
    notices = (["目标使用无 Embedding 的 AGFS 检索，不创建模型或向量索引；导入后按目标配置重新建立全文缓存。"]
               if vector_free else ["始终使用目标模型重建向量，不复用源向量；来源描述不会覆盖目标配置。"])
    if source is None:
        notices.append("迁移包缺少来源环境描述，无法比较模型、维度及数据库版本。")
    elif any(name.startswith("embedding.") for name in changes):
        notices.append("Embedding 配置发生变化，重建后的检索结果可能变化。")
    if not vector_free and (target["embedding"]["dimension"] is None or (
        source is not None and source["embedding"]["dimension"] is None
    )):
        notices.append("部分模型维度无法获取，显示为未知；不能据此判断模型维度相同。")
    if not vector_free and target["embedding"]["dimension"] is None:
        notices.append("目标模型维度尚未验证，导入前仍须通过模型探测与索引维度校验。")
    if "content_storage.backend" in changes:
        notices.append("内容存储发生变化，将按目标存储适配器恢复并校验记录。")
    if "content_storage.database_type" in changes:
        notices.append("SQL 数据库类型发生变化，将按目标数据库约束校验并恢复。")
    if "vector_storage.backend" in changes:
        notices.append("向量后端发生变化，将按目标检索模式恢复索引。")
    return {"source": source, "target": target, "changes": changes, "notices": notices}
