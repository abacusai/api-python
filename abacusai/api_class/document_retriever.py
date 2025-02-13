import dataclasses

from .abstract import ApiClass
from .enums import VectorStoreTextEncoder


@dataclasses.dataclass
class VectorStoreConfig(ApiClass):
    """
    Config for indexing options of a document retriever. Default values of optional arguments are heuristically selected by the Abacus.AI platform based on the underlying data.

    Args:
        chunk_size (int): The size of text chunks in the vector store.
        chunk_overlap_fraction (float): The fraction of overlap between chunks.
        text_encoder (VectorStoreTextEncoder): Encoder used to index texts from the documents.
        chunk_size_factors (list): Chunking data with multiple sizes. The specified list of factors are used to calculate more sizes, in addition to `chunk_size`.
        score_multiplier_column (str): If provided, will use the values in this metadata column to modify the relevance score of returned chunks for all queries.
        prune_vectors (bool): Transform vectors using SVD so that the average component of vectors in the corpus are removed.
        index_metadata_columns (bool): If True, metadata columns of the FG will also be used for indexing and querying.
        use_document_summary (bool): If True, uses the summary of the document in addition to chunks of the document for indexing and querying.
        summary_instructions (str): Instructions for the LLM to generate the document summary.
        standalone_deployment (bool): If True, the document retriever will be deployed as a standalone deployment.
    """
    chunk_size: int = dataclasses.field(default=None)
    chunk_overlap_fraction: float = dataclasses.field(default=None)
    text_encoder: VectorStoreTextEncoder = dataclasses.field(default=None)
    chunk_size_factors: list = dataclasses.field(default=None)
    score_multiplier_column: str = dataclasses.field(default=None)
    prune_vectors: bool = dataclasses.field(default=None)
    index_metadata_columns: bool = dataclasses.field(default=None)
    use_document_summary: bool = dataclasses.field(default=None)
    summary_instructions: str = dataclasses.field(default=None)
    standalone_deployment: bool = dataclasses.field(default=False)


DocumentRetrieverConfig = VectorStoreConfig
