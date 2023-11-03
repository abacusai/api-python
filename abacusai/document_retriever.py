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
            featureGroupId (str): The feature group id associated with the document retriever.
            featureGroupName (str): The feature group name associated with the document retriever.
            latestDocumentRetrieverVersion (DocumentRetrieverVersion): The latest version of vector store.
            documentRetrieverConfig (DocumentRetrieverConfig): The config for vector store creation.
    """

    def __init__(self, client, name=None, documentRetrieverId=None, createdAt=None, featureGroupId=None, featureGroupName=None, latestDocumentRetrieverVersion={}, documentRetrieverConfig={}):
        super().__init__(client, documentRetrieverId)
        self.name = name
        self.document_retriever_id = documentRetrieverId
        self.created_at = createdAt
        self.feature_group_id = featureGroupId
        self.feature_group_name = featureGroupName
        self.latest_document_retriever_version = client._build_class(
            DocumentRetrieverVersion, latestDocumentRetrieverVersion)
        self.document_retriever_config = client._build_class(
            DocumentRetrieverConfig, documentRetrieverConfig)

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'document_retriever_id': repr(self.document_retriever_id), f'created_at': repr(self.created_at), f'feature_group_id': repr(self.feature_group_id), f'feature_group_name': repr(
            self.feature_group_name), f'latest_document_retriever_version': repr(self.latest_document_retriever_version), f'document_retriever_config': repr(self.document_retriever_config)}
        class_name = "DocumentRetriever"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'document_retriever_id': self.document_retriever_id, 'created_at': self.created_at, 'feature_group_id': self.feature_group_id, 'feature_group_name': self.feature_group_name,
                'latest_document_retriever_version': self._get_attribute_as_dict(self.latest_document_retriever_version), 'document_retriever_config': self._get_attribute_as_dict(self.document_retriever_config)}
        return {key: value for key, value in resp.items() if value is not None}

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
            list[DocumentRetrieverVersion]: All the document retriever versions associated with the document retriever.
        """
        return self.client.list_document_retriever_versions(self.document_retriever_id, limit, start_after_version)

    def get_document_snippet(self, document_id: str, start_word_index: int = None, end_word_index: int = None):
        """
        Get a snippet from documents in the document retriever.

        Args:
            document_id (str): The ID of the document to retrieve the snippet from.
            start_word_index (int): If provided, will start the snippet at the index (of words in the document) specified.
            end_word_index (int): If provided, will end the snippet at the index of (of words in the document) specified.

        Returns:
            DocumentRetrieverLookupResult: The documentation snippet found from the document retriever.
        """
        return self.client.get_document_snippet(self.document_retriever_id, document_id, start_word_index, end_word_index)

    def restart(self):
        """
        Restart the document retriever if it is stopped.

        Args:
            document_retriever_id (str): A unique string identifier associated with the document retriever.
        """
        return self.client.restart_document_retriever(self.document_retriever_id)

    def wait_until_ready(self, timeout: int = 3600):
        """
        A waiting call until document retriever is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 3600 seconds.
        """
        if self.get_status() == 'STOPPED':
            self.client.restart_document_retriever(self.document_retriever_id)
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

    def get_matching_documents(self, query: str, filters: dict = None, limit: int = None, result_columns: list = None, max_words: int = None, num_retrieval_margin_words: int = None, max_words_per_chunk: int = None):
        """
        Lookup document retrievers and return the matching documents from the document retriever deployed with given query.

        Original documents are splitted into chunks and stored in the document retriever. This lookup function will return the relevant chunks
        from the document retriever. The returned chunks could be expanded to include more words from the original documents and merged if they
        are overlapping, and permitted by the settings provided. The returned chunks are sorted by relevance.


        Args:
            query (str): The query to search for.
            filters (dict): A dictionary mapping column names to a list of values to restrict the retrieved search results.
            limit (int): If provided, will limit the number of results to the value specified.
            result_columns (list): If provided, will limit the column properties present in each result to those specified in this list.
            max_words (int): If provided, will limit the total number of words in the results to the value specified.
            num_retrieval_margin_words (int): If provided, will add this number of words from left and right of the returned chunks.
            max_words_per_chunk (int): If provided, will limit the number of words in each chunk to the value specified. If the value provided is smaller than the actual size of chunk on disk, which is determined during document retriever creation, the actual size of chunk will be used. I.e, chunks looked up from document retrievers will not be split into smaller chunks during lookup due to this setting.

        Returns:
            list[DocumentRetrieverLookupResult]: The relevant documentation results found from the document retriever.
        """
        return self.client.get_matching_documents(self.document_retriever_id, query, filters, limit, result_columns, max_words, num_retrieval_margin_words, max_words_per_chunk)
