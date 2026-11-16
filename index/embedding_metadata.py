"""Shared embedding payload and model identity for index writers."""

from dataclasses import replace


def augment_with_embeddings(records, embeddings, embedder):
    augmented = []
    model = getattr(embedder, "model", None)
    for record, embedding in zip(records, embeddings):
        metadata = {
            **record.metadata,
            "_embedding": embedding,
            "embedding_dimension": len(embedding),
        }
        if isinstance(model, str) and model:
            metadata["embedding_model"] = model
        augmented.append(replace(record, metadata=metadata))
    return augmented
