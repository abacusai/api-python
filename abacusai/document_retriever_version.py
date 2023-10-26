from .api_class import DocumentRetrieverConfig
from .document_retriever_config import DocumentRetrieverConfig
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
            error (str): The error message when it failed to create the document retriever version.
            numberOfChunks (int): The number of chunks for the document retriever.
            embeddingFileSize (int): The size of embedding file for the document retriever.
            warnings (list): (list): The warning messages when creating the document retriever.
            resolvedConfig (DocumentRetrieverConfig): The resolved configurations, such as default settings, for indexing documents.
    """

    def __init__(self, client, documentRetrieverId=None, documentRetrieverVersion=None, createdAt=None, status=None, featureGroupVersion=None, error=None, numberOfChunks=None, embeddingFileSize=None, warnings=None, resolvedConfig={}):
        super().__init__(client, documentRetrieverVersion)
        self.document_retriever_id = documentRetrieverId
        self.document_retriever_version = documentRetrieverVersion
        self.created_at = createdAt
        self.status = status
        self.feature_group_version = featureGroupVersion
        self.error = error
        self.number_of_chunks = numberOfChunks
        self.embedding_file_size = embeddingFileSize
        self.warnings = warnings
        self.resolved_config = client._build_class(
            DocumentRetrieverConfig, resolvedConfig)

    def __repr__(self):
        repr_dict = {f'document_retriever_id': repr(self.document_retriever_id), f'document_retriever_version': repr(self.document_retriever_version), f'created_at': repr(self.created_at), f'status': repr(self.status), f'feature_group_version': repr(
            self.feature_group_version), f'error': repr(self.error), f'number_of_chunks': repr(self.number_of_chunks), f'embedding_file_size': repr(self.embedding_file_size), f'warnings': repr(self.warnings), f'resolved_config': repr(self.resolved_config)}
        class_name = "DocumentRetrieverVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'document_retriever_id': self.document_retriever_id, 'document_retriever_version': self.document_retriever_version, 'created_at': self.created_at, 'status': self.status, 'feature_group_version': self.feature_group_version,
                'error': self.error, 'number_of_chunks': self.number_of_chunks, 'embedding_file_size': self.embedding_file_size, 'warnings': self.warnings, 'resolved_config': self._get_attribute_as_dict(self.resolved_config)}
        return {key: value for key, value in resp.items() if value is not None}

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
        return self.client._poll(self, {'PENDING', 'INDEXING', 'DEPLOYING'}, timeout=timeout)

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
