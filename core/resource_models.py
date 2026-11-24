"""Domain models for resources and their retrieval results."""

from dataclasses import dataclass
from typing import ClassVar
from uuid import UUID

from core.enums import ContextType
from core.resource_identity import validate_resource_uri_slug


@dataclass(frozen=True)
class Resource:
    account_id: str
    resource_id: UUID

    context_type: ClassVar[ContextType] = ContextType.RESOURCE

    def __post_init__(self) -> None:
        validate_resource_uri_slug(self.account_id)


@dataclass(frozen=True)
class DocumentResource(Resource):
    filename: str
    media_type: str

    resource_type: ClassVar[str] = "document"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not self.filename.isprintable()
            or self.filename in {".", ".."}
            or any(char in self.filename for char in ("/", "\\"))
        ):
            raise ValueError("filename must be a printable basename")
        if not self.media_type.isprintable():
            raise ValueError("media_type must be printable")

    @property
    def uri(self) -> str:
        return f"ctx://{self.account_id}/resources/doc-{self.resource_id.hex}"


@dataclass(frozen=True)
class ResourceChunk:
    uri: str
    chunk_id: str
    content: str
    rank: int
