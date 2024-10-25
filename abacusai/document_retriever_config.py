from .api_class import DocumentRetrieverConfig
from .return_class import AbstractApiClass


class DocumentRetrieverConfig(AbstractApiClass):
    """
        A config for document retriever creation.

        Args:
            client (ApiClient): An authenticated API Client instance
            chunkSize (int): The size of chunks for vector store, i.e., maximum number of words in the chunk.
            chunkOverlapFraction (float): The fraction of overlap between two consecutive chunks.
            textEncoder (str): The text encoder used to encode texts in the vector store.
            scoreMultiplierColumn (str): The values in this metadata column are used to modify the relevance scores of returned chunks.
            pruneVectors (bool): Corpus specific transformation of vectors that applies dimensional reduction techniques to strip common components from the vectors.
            indexMetadataColumns (bool): If True, metadata columns of the FG will also be used for indexing and querying.
            useDocumentSummary (bool): If True, uses the summary of the document in addition to chunks of the document for indexing and querying.
            summaryInstructions (str): Instructions for the LLM to generate the document summary.
    """

    def __init__(self, client, chunkSize=None, chunkOverlapFraction=None, textEncoder=None, scoreMultiplierColumn=None, pruneVectors=None, indexMetadataColumns=None, useDocumentSummary=None, summaryInstructions=None):
        super().__init__(client, None)
        self.chunk_size = chunkSize
        self.chunk_overlap_fraction = chunkOverlapFraction
        self.text_encoder = textEncoder
        self.score_multiplier_column = scoreMultiplierColumn
        self.prune_vectors = pruneVectors
        self.index_metadata_columns = indexMetadataColumns
        self.use_document_summary = useDocumentSummary
        self.summary_instructions = summaryInstructions
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chunk_size': repr(self.chunk_size), f'chunk_overlap_fraction': repr(self.chunk_overlap_fraction), f'text_encoder': repr(self.text_encoder), f'score_multiplier_column': repr(self.score_multiplier_column), f'prune_vectors': repr(
            self.prune_vectors), f'index_metadata_columns': repr(self.index_metadata_columns), f'use_document_summary': repr(self.use_document_summary), f'summary_instructions': repr(self.summary_instructions)}
        class_name = "DocumentRetrieverConfig"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chunk_size': self.chunk_size, 'chunk_overlap_fraction': self.chunk_overlap_fraction, 'text_encoder': self.text_encoder, 'score_multiplier_column': self.score_multiplier_column,
                'prune_vectors': self.prune_vectors, 'index_metadata_columns': self.index_metadata_columns, 'use_document_summary': self.use_document_summary, 'summary_instructions': self.summary_instructions}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
