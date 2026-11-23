# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Text embedding adapter for the Volcengine Ark protocol."""

import logging
from typing import Final

from core.interfaces import Embedder
from providers.token_tracker import TokenTracker, embedding_input_scope, embedding_usage_tokens

logger = logging.getLogger(__name__)

DEFAULT_MODEL: Final[str] = "doubao-embedding-vision-250615"
DEFAULT_DIMENSION: Final[int] = 1024
DEFAULT_BASE_URL: Final[str] = "https://ark.cn-beijing.volces.com/api/coding/v3"


class VolcengineEmbedder(Embedder):
    """Embed text inputs through the existing Ark embedding adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimension: int = DEFAULT_DIMENSION,
    ):
        try:
            import volcenginesdkarkruntime
        except ImportError as exc:
            raise ImportError(
                "Volcengine text embeddings require volcengine-ark-runtime. "
                "Install with: pip install volcengine-ark-runtime"
            ) from exc

        self._model_name = model
        self._dimension = dimension
        self._client = volcenginesdkarkruntime.Ark(api_key=api_key, base_url=base_url)
        self.token_tracker = TokenTracker(model=self._model_name)
        logger.info("VolcengineEmbedder loaded: model=%s, dim=%d", model, dimension)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed each text as one document and retain its usage attribution."""
        if not texts:
            return []

        results: list[list[float]] = []
        for index, text in enumerate(texts):
            try:
                response = self._client.multimodal_embeddings.create(
                    input=[{"type": "text", "text": text}],
                    model=self._model_name,
                    dimensions=self._dimension,
                )
            except Exception:
                with embedding_input_scope([index]):
                    self.token_tracker.record_failure('embedding')
                raise
            total = embedding_usage_tokens(getattr(response, 'usage', None))
            with embedding_input_scope([index]):
                self.token_tracker.record_embed(total or 0, usage_known=total is not None)
            data = response.data
            if hasattr(data, "embedding"):
                results.append(data.embedding)
            elif isinstance(data, list) and data:
                results.append(data[0].embedding)
            else:
                raise ValueError(f"Unexpected embedding response for text: {text[:50]}")
        return results

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model_name
