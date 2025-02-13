from .api_class import VectorStoreConfig
from .return_class import AbstractApiClass


class DocumentRetrieverVersion(AbstractApiClass):
    """
        A version of document retriever.

        Args:
            client (ApiClient): An authenticated API Client instance
            documentRetrieverId (str): The unique identifier of the Document Retriever.
            documentRetrieverVersion (str): The unique identifier of the Document Retriever version.
            createdAt (str): When the Document Retriever was created.
            status (str): The status of Document Retriever version. It represents indexing status until indexing isn't complete, and deployment status after indexing is complete.
            deploymentStatus (str): The status of deploying the Document Retriever version.
            featureGroupId (str): The feature group id associated with the document retriever.
            featureGroupVersion (str): The unique identifier of the feature group version at which the Document Retriever version is created.
            error (str): The error message when it failed to create the document retriever version.
            numberOfChunks (int): The number of chunks for the document retriever.
            embeddingFileSize (int): The size of embedding file for the document retriever.
            warnings (list): The warning messages when creating the document retriever.
            resolvedConfig (VectorStoreConfig): The resolved configurations, such as default settings, for indexing documents.
            documentRetrieverConfig (VectorStoreConfig): The config used to create the document retriever version.
    """

    def __init__(self, client, documentRetrieverId=None, documentRetrieverVersion=None, createdAt=None, status=None, deploymentStatus=None, featureGroupId=None, featureGroupVersion=None, error=None, numberOfChunks=None, embeddingFileSize=None, warnings=None, resolvedConfig={}, documentRetrieverConfig={}):
        super().__init__(client, documentRetrieverVersion)
        self.document_retriever_id = documentRetrieverId
        self.document_retriever_version = documentRetrieverVersion
        self.created_at = createdAt
        self.status = status
        self.deployment_status = deploymentStatus
        self.feature_group_id = featureGroupId
        self.feature_group_version = featureGroupVersion
        self.error = error
        self.number_of_chunks = numberOfChunks
        self.embedding_file_size = embeddingFileSize
        self.warnings = warnings
        self.resolved_config = client._build_class(
            VectorStoreConfig, resolvedConfig)
        self.document_retriever_config = client._build_class(
            VectorStoreConfig, documentRetrieverConfig)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'document_retriever_id': repr(self.document_retriever_id), f'document_retriever_version': repr(self.document_retriever_version), f'created_at': repr(self.created_at), f'status': repr(self.status), f'deployment_status': repr(self.deployment_status), f'feature_group_id': repr(self.feature_group_id), f'feature_group_version': repr(
            self.feature_group_version), f'error': repr(self.error), f'number_of_chunks': repr(self.number_of_chunks), f'embedding_file_size': repr(self.embedding_file_size), f'warnings': repr(self.warnings), f'resolved_config': repr(self.resolved_config), f'document_retriever_config': repr(self.document_retriever_config)}
        class_name = "DocumentRetrieverVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'document_retriever_id': self.document_retriever_id, 'document_retriever_version': self.document_retriever_version, 'created_at': self.created_at, 'status': self.status, 'deployment_status': self.deployment_status, 'feature_group_id': self.feature_group_id, 'feature_group_version': self.feature_group_version,
                'error': self.error, 'number_of_chunks': self.number_of_chunks, 'embedding_file_size': self.embedding_file_size, 'warnings': self.warnings, 'resolved_config': self._get_attribute_as_dict(self.resolved_config), 'document_retriever_config': self._get_attribute_as_dict(self.document_retriever_config)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def delete(self):
        """
        Delete a document retriever version.

        Args:
            document_retriever_version (str): A unique string identifier associated with the document retriever version.
        """
        return self.client.delete_document_retriever_version(self.document_retriever_version)

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
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        self.client._poll(self, {'PENDING', 'INDEXING'}, timeout=timeout / 2)
        if self.get_deployment_status() == 'STOPPED':
            self.client.restart_document_retriever(self.document_retriever_id)
        self.wait_until_deployment_ready(timeout=timeout / 2)
        return self.refresh()

    def wait_until_ready(self, timeout=3600):
        """
        A waiting call until the document retriever version is ready.  It restarts the document retriever if it is stopped.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_results(timeout)

    def wait_until_deployment_ready(self, timeout: int = 3600):
        """
        A waiting call until the document retriever deployment is ready to serve.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 3600 seconds.
        """
        import time

        from .client import ApiException
        start_time = time.time()
        while (True):
            deployment_status = self.get_deployment_status()
            if deployment_status in {'PENDING', 'DEPLOYING'}:
                if timeout and time.time() - start_time > timeout:
                    raise TimeoutError(
                        f'Maximum wait time of {timeout}s exceeded')
            elif deployment_status in {'STOPPED'}:
                raise ApiException(
                    409, f'Document retriever deployment is stopped, please restart it.')
            elif deployment_status in {'FAILED'}:
                raise ApiException(
                    409, f'Document retriever deployment failed, please retry deploying it.')
            else:
                return self.refresh()
            time.sleep(15)

    def get_status(self):
        """
        Gets the status of the document retriever version.

        Returns:
            str: A string describing the status of a document retriever version (pending, complete, etc.).
        """
        return self.describe().status

    def get_deployment_status(self):
        """
        Gets the status of the document retriever version.

        Returns:
            str: A string describing the deployment status of a document retriever version (pending, deploying, etc.).
        """
        return self.describe().deployment_status
