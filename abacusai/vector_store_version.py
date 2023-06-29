from .return_class import AbstractApiClass


class VectorStoreVersion(AbstractApiClass):
    """
        A version of vector store.

        Args:
            client (ApiClient): An authenticated API Client instance
            vectorStoreId (str): The unique identifier of the vector store.
            vectorStoreVersion (str): The unique identifier of the vector store version.
            createdAt (str): When the vector store was created.
            status (str): The status of creating vector store version.
            featureGroupVersion (str): The unique identifier of the feature group version at which the vector store version is created.
    """

    def __init__(self, client, vectorStoreId=None, vectorStoreVersion=None, createdAt=None, status=None, featureGroupVersion=None):
        super().__init__(client, vectorStoreVersion)
        self.vector_store_id = vectorStoreId
        self.vector_store_version = vectorStoreVersion
        self.created_at = createdAt
        self.status = status
        self.feature_group_version = featureGroupVersion

    def __repr__(self):
        return f"VectorStoreVersion(vector_store_id={repr(self.vector_store_id)},\n  vector_store_version={repr(self.vector_store_version)},\n  created_at={repr(self.created_at)},\n  status={repr(self.status)},\n  feature_group_version={repr(self.feature_group_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'vector_store_id': self.vector_store_id, 'vector_store_version': self.vector_store_version, 'created_at': self.created_at, 'status': self.status, 'feature_group_version': self.feature_group_version}

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until vector store version indexing and deployment is complete.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'INDEXING'}, timeout=timeout)

    def wait_until_ready(self, timeout=3600):
        """
        A waiting call until the vector store version is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_results(timeout)

    def get_status(self):
        """
        Gets the status of the vector store version.

        Returns:
            str: A string describing the status of a vector store version (pending, complete, etc.).
        """
        return self.describe().status
