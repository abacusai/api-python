from .return_class import AbstractApiClass


class DocumentRetrieverVersion(AbstractApiClass):
    """
        A version of document retriever.

        Args:
            client (ApiClient): An authenticated API Client instance
            documentRetrieverId (str): The unique identifier of the vector store.
            documentRetrieverVersion (str): The unique identifier of the vector store version.
            createdAt (str): When the vector store was created.
            status (str): The status of creating vector store version.
            featureGroupVersion (str): The unique identifier of the feature group version at which the vector store version is created.
    """

    def __init__(self, client, documentRetrieverId=None, documentRetrieverVersion=None, createdAt=None, status=None, featureGroupVersion=None):
        super().__init__(client, documentRetrieverVersion)
        self.document_retriever_id = documentRetrieverId
        self.document_retriever_version = documentRetrieverVersion
        self.created_at = createdAt
        self.status = status
        self.feature_group_version = featureGroupVersion

    def __repr__(self):
        return f"DocumentRetrieverVersion(document_retriever_id={repr(self.document_retriever_id)},\n  document_retriever_version={repr(self.document_retriever_version)},\n  created_at={repr(self.created_at)},\n  status={repr(self.status)},\n  feature_group_version={repr(self.feature_group_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'document_retriever_id': self.document_retriever_id, 'document_retriever_version': self.document_retriever_version, 'created_at': self.created_at, 'status': self.status, 'feature_group_version': self.feature_group_version}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            DocumentRetrieverVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describe a document retriever version.

        Args:
            document_retriever_version (str): A unique string identifier associated with the document retriever version.

        Returns:
            DocumentRetrieverVersion: The document retriever version object.
        """
        return self.client.describe_document_retriever_version(self.document_retriever_version)

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until document retriever version is complete.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'INDEXING'}, timeout=timeout)

    def wait_until_ready(self, timeout=3600):
        """
        A waiting call until the document retriever version is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_results(timeout)

    def get_status(self):
        """
        Gets the status of the document retriever version.

        Returns:
            str: A string describing the status of a document retriever version (pending, complete, etc.).
        """
        return self.describe().status
