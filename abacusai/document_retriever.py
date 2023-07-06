from typing import Union

from .api_class import DocumentRetrieverConfig
from .document_retriever_config import DocumentRetrieverConfig
from .document_retriever_version import DocumentRetrieverVersion
from .return_class import AbstractApiClass


class DocumentRetriever(AbstractApiClass):
    """
        A vector store that stores embeddings for a list of document trunks.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the document retriever.
            documentRetrieverId (str): The unique identifier of the vector store.
            createdAt (str): When the vector store was created.
            latestDocumentRetrieverVersion (DocumentRetrieverVersion): The latest version of vector store.
            documentRetrieverConfig (DocumentRetrieverConfig): The config for vector store creation.
    """

    def __init__(self, client, name=None, documentRetrieverId=None, createdAt=None, latestDocumentRetrieverVersion={}, documentRetrieverConfig={}):
        super().__init__(client, documentRetrieverId)
        self.name = name
        self.document_retriever_id = documentRetrieverId
        self.created_at = createdAt
        self.latest_document_retriever_version = client._build_class(
            DocumentRetrieverVersion, latestDocumentRetrieverVersion)
        self.document_retriever_config = client._build_class(
            DocumentRetrieverConfig, documentRetrieverConfig)

    def __repr__(self):
        return f"DocumentRetriever(name={repr(self.name)},\n  document_retriever_id={repr(self.document_retriever_id)},\n  created_at={repr(self.created_at)},\n  latest_document_retriever_version={repr(self.latest_document_retriever_version)},\n  document_retriever_config={repr(self.document_retriever_config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'document_retriever_id': self.document_retriever_id, 'created_at': self.created_at, 'latest_document_retriever_version': self._get_attribute_as_dict(self.latest_document_retriever_version), 'document_retriever_config': self._get_attribute_as_dict(self.document_retriever_config)}

    def update(self, name: str = None, feature_group_id: str = None, document_retriever_config: Union[dict, DocumentRetrieverConfig] = None):
        """
        Updates an existing document retriever.

        Args:
            name (str): The name group to update the document retriever with.
            feature_group_id (str): The ID of the feature group to update the document retriever with.
            document_retriever_config (DocumentRetrieverConfig): The configuration, including chunk_size and chunk_overlap_fraction, for document retrieval.

        Returns:
            DocumentRetriever: The updated document retriever.
        """
        return self.client.update_document_retriever(self.document_retriever_id, name, feature_group_id, document_retriever_config)

    def create_version(self):
        """
        Creates a document retriever version from the latest version of the feature group that the document retriever associated with.

        Args:
            document_retriever_id (str): The unique ID associated with the document retriever to create version with.

        Returns:
            DocumentRetrieverVersion: The newly created document retriever version.
        """
        return self.client.create_document_retriever_version(self.document_retriever_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            DocumentRetriever: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describe a Document Retriever.

        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever.

        Returns:
            DocumentRetriever: The document retriever object.
        """
        return self.client.describe_document_retriever(self.document_retriever_id)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        List all the document retriever versions with a given ID.

        Args:
            limit (int): The number of vector store versions to retrieve.
            start_after_version (str): An offset parameter to exclude all document retriever versions up to this specified one.

        Returns:
            DocumentRetrieverVersion: All the document retriever versions associated with the document retriever.
        """
        return self.client.list_document_retriever_versions(self.document_retriever_id, limit, start_after_version)

    def lookup(self, query: str, deployment_token: str, limit_results: int = None):
        """
        Lookup relevant documents from the document retriever deployed with given query.

        Args:
            query (str): The query to search for.
            deployment_token (str): A deployment token used to authenticate access to created vector store.
            limit_results (int): If provided, will limit the number of results to the value specified.

        Returns:
            DocumentRetrieverLookupResult: The relevant documentation results found from the document retriever.
        """
        return self.client.lookup_document_retriever(self.document_retriever_id, query, deployment_token, limit_results)

    def wait_until_ready(self, timeout: int = 3600):
        """
        A waiting call until document retriever is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 3600 seconds.
        """
        version = self.describe().latest_document_retriever_version
        if not version:
            from .client import ApiException
            raise ApiException(
                409, 'This vector store does not have any versions')
        version.wait_until_ready(timeout=timeout)
        return self

    def get_status(self):
        """
        Gets the status of the document retriever.

        Returns:
            str: A string describing the status of a document retriever (pending, complete, etc.).
        """
        return self.describe().latest_document_retriever_version.status
