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
        chunk_size_factors (list): Chunking data with multiple sizes. The specified list of factors are used to calculate more sizes, in addition to `chunk_size`.
        score_multiplier_column (str): If provided, will use the values in this metadata column to modify the relevance score of returned chunks for all queries.
        prune_vectors (bool): Transform vectors using SVD so that the average component of vectors in the corpus are removed.
    """
    chunk_size: int = dataclasses.field(default=None)
    chunk_overlap_fraction: float = dataclasses.field(default=None)
    text_encoder: VectorStoreTextEncoder = dataclasses.field(default=None)
    chunk_size_factors: list = dataclasses.field(default=None)
    score_multiplier_column: str = dataclasses.field(default=None)
    prune_vectors: bool = dataclasses.field(default=None)


@dataclasses.dataclass
class DocumentRetrieverConfig(VectorStoreConfig):
    """
    Configs for document retriever. If any configuration value is not explicitly provided, Abacus will automatically infer default values based on the data.
    """
