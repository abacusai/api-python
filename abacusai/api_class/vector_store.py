import dataclasses

from .abstract import ApiClass
from .enums import VectorStoreTextEncoder


@dataclasses.dataclass
class VectorStoreConfig(ApiClass):
    """
    Configs for vector store indexing.

    Args:
        chunk_size (int): The size of text chunks in the vector store.
        chunk_overlap_fraction (float): The fraction of overlap between chunks.
    """
    chunk_size: int = dataclasses.field(default=512)
    chunk_overlap_fraction: float = dataclasses.field(default=0.1)
    text_encoder: VectorStoreTextEncoder = dataclasses.field(default=VectorStoreTextEncoder.E5)


@dataclasses.dataclass
class DocumentRetrieverConfig(VectorStoreConfig):
    """
    Configs for document retriever.
    """
