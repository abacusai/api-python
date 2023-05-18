from .return_class import AbstractApiClass


class VectorStore(AbstractApiClass):
    """
        A vector store that stores embeddings for a list of document trunks.

        Args:
            client (ApiClient): An authenticated API Client instance
            vectorStoreId (str): The unique identifier of the vector store.
            createdAt (str): When the vector store was created.
    """

    def __init__(self, client, vectorStoreId=None, createdAt=None):
        super().__init__(client, vectorStoreId)
        self.vector_store_id = vectorStoreId
        self.created_at = createdAt

    def __repr__(self):
        return f"VectorStore(vector_store_id={repr(self.vector_store_id)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'vector_store_id': self.vector_store_id, 'created_at': self.created_at}
