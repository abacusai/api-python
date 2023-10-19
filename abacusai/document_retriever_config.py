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
    """

    def __init__(self, client, chunkSize=None, chunkOverlapFraction=None, textEncoder=None):
        super().__init__(client, None)
        self.chunk_size = chunkSize
        self.chunk_overlap_fraction = chunkOverlapFraction
        self.text_encoder = textEncoder

    def __repr__(self):
        repr_dict = {f'chunk_size': repr(self.chunk_size), f'chunk_overlap_fraction': repr(
            self.chunk_overlap_fraction), f'text_encoder': repr(self.text_encoder)}
        class_name = "DocumentRetrieverConfig"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chunk_size': self.chunk_size, 'chunk_overlap_fraction':
                self.chunk_overlap_fraction, 'text_encoder': self.text_encoder}
        return {key: value for key, value in resp.items() if value is not None}
