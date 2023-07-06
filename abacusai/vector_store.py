from .api_class import VectorStoreConfig
from .return_class import AbstractApiClass
from .vector_store_config import VectorStoreConfig
from .vector_store_version import VectorStoreVersion


class VectorStore(AbstractApiClass):
    """
        A vector store that stores embeddings for a list of document trunks.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the vector store.
            vectorStoreId (str): The unique identifier of the vector store.
            createdAt (str): When the vector store was created.
            latestVectorStoreVersion (VectorStoreVersion): The latest version of vector store.
            vectorStoreConfig (VectorStoreConfig): The config for vector store creation.
    """

    def __init__(self, client, name=None, vectorStoreId=None, createdAt=None, latestVectorStoreVersion={}, vectorStoreConfig={}):
        super().__init__(client, vectorStoreId)
        self.name = name
        self.vector_store_id = vectorStoreId
        self.created_at = createdAt
        self.latest_vector_store_version = client._build_class(
            VectorStoreVersion, latestVectorStoreVersion)
        self.vector_store_config = client._build_class(
            VectorStoreConfig, vectorStoreConfig)

    def __repr__(self):
        return f"VectorStore(name={repr(self.name)},\n  vector_store_id={repr(self.vector_store_id)},\n  created_at={repr(self.created_at)},\n  latest_vector_store_version={repr(self.latest_vector_store_version)},\n  vector_store_config={repr(self.vector_store_config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'vector_store_id': self.vector_store_id, 'created_at': self.created_at, 'latest_vector_store_version': self._get_attribute_as_dict(self.latest_vector_store_version), 'vector_store_config': self._get_attribute_as_dict(self.vector_store_config)}

    def delete_document_retriever(self):
        """
        Delete a Document Retriever.

        Args:
            vector_store_id (str): A unique string identifier associated with the document retriever.
        """
        return self.client.delete_document_retriever(self.vector_store_id)

    def wait_until_ready(self, timeout: int = 3600):
        """
        A waiting call until vector store is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 3600 seconds.
        """
        version = self.describe().latest_vector_store_version
        if not version:
            from .client import ApiException
            raise ApiException(
                409, 'This vector store does not have any versions')
        version.wait_until_ready(timeout=timeout)
        return self

    def get_status(self):
        """
        Gets the status of the vector store.

        Returns:
            str: A string describing the status of a vector store (pending, complete, etc.).
        """
        return self.describe().latest_vector_store_version.status
