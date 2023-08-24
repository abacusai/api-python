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
        text_encoder (VectorStoreTextEncoder): Encoder used to index texts from the documents.
    """
    chunk_size: int = dataclasses.field(default=None)
    chunk_overlap_fraction: float = dataclasses.field(default=None)
    text_encoder: VectorStoreTextEncoder = dataclasses.field(default=None)


@dataclasses.dataclass
class DocumentRetrieverConfig(VectorStoreConfig):
    """
    Configs for document retriever.
    """
