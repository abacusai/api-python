from .return_class import AbstractApiClass
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
    """

    def __init__(self, client, name=None, vectorStoreId=None, createdAt=None, latestVectorStoreVersion={}):
        super().__init__(client, vectorStoreId)
        self.name = name
        self.vector_store_id = vectorStoreId
        self.created_at = createdAt
        self.latest_vector_store_version = client._build_class(
            VectorStoreVersion, latestVectorStoreVersion)

    def __repr__(self):
        return f"VectorStore(name={repr(self.name)},\n  vector_store_id={repr(self.vector_store_id)},\n  created_at={repr(self.created_at)},\n  latest_vector_store_version={repr(self.latest_vector_store_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'vector_store_id': self.vector_store_id, 'created_at': self.created_at, 'latest_vector_store_version': self._get_attribute_as_dict(self.latest_vector_store_version)}

    def update(self, feature_group_id: str = None, name: str = None):
        """
        Updates an existing vector store.

        Args:
            feature_group_id (str): The ID of the feature group to update the vector store with.
            name (str): The name group to update the vector store with.

        Returns:
            VectorStore: The updated vector store.
        """
        return self.client.update_vector_store(self.vector_store_id, feature_group_id, name)

    def create_version(self):
        """
        Creates a vector store version from the latest version of the feature group that the vector store associated with.

        Args:
            vector_store_id (str): The unique ID associated with the vector store to create version with.

        Returns:
            VectorStoreVersion: The newly created vector store version.
        """
        return self.client.create_vector_store_version(self.vector_store_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            VectorStore: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describe a Vector Store.

        Args:
            vector_store_id (str): A unique string identifier associated with the vector store.

        Returns:
            VectorStore: The vector store object.
        """
        return self.client.describe_vector_store(self.vector_store_id)

    def delete(self):
        """
        Delete a Vector Store.

        Args:
            vector_store_id (str): A unique string identifier associated with the vector store.
        """
        return self.client.delete_vector_store(self.vector_store_id)

    def list_versions(self):
        """
        List all the vector store versions with a given vector store ID.

        Args:
            vector_store_id (str): A unique string identifier associated with the vector store.

        Returns:
            VectorStoreVersion: All the vector store versions associated with the vector store.
        """
        return self.client.list_vector_store_versions(self.vector_store_id)

    def lookup(self, query: str, deployment_token: str, limit_results: int = None):
        """
        Lookup relevant documents from the vector store deployed with given query.

        Args:
            query (str): The query to search for.
            deployment_token (str): A deployment token used to authenticate access to created vector store.
            limit_results (int): If provided, will limit the number of results to the value specified.

        Returns:
            VectorStoreLookupResult: The relevant documentation results found from the vector store.
        """
        return self.client.lookup_vector_store(self.vector_store_id, query, deployment_token, limit_results)

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
