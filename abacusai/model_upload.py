from .return_class import AbstractApiClass


class ModelUpload(AbstractApiClass):
    """
        A model version that includes the upload identifiers for the various required files.

        Args:
            client (ApiClient): An authenticated API Client instance
            modelId (str): A reference to the model this version belongs to.
            modelVersion (str): A unique identifier for the model version.
            status (str): The current status of the model.
            createdAt (str): The timestamp at which the model version was created, in ISO-8601 format.
            modelUploadId (str): An upload identifier to be used when uploading the TensorFlow Saved Model.
            embeddingsUploadId (str): An upload identifier to be used when uploading the embeddings CSV.
            artifactsUploadId (str): An upload identifier to be used when uploading the artifacts JSON file.
            verificationsUploadId (str): An upload identifier to be used when uploading the verifications JSON file.
            defaultItemsUploadId (str): An upload identifier to be used when uploading the default items JSON file.
            modelFileUploadId (str): An upload identifier to be used when uploading the model JSON file.
            modelStateUploadId (str): An upload identifier to be used when uploading the model state JSON file.
            inputPreprocessorUploadId (str): An upload identifier to be used when uploading the input preprocessor JSON file.
            requirementsUploadId (str): An upload identifier to be used when uploading the requirements JSON file.
            resourcesUploadId (str): An upload identifier to be used when uploading the resources JSON file.
            multiCatalogEmbeddingsUploadId (str): An upload identifier to be used when upload the multi-catalog embeddings CSV file.
    """

    def __init__(self, client, modelId=None, modelVersion=None, status=None, createdAt=None, modelUploadId=None, embeddingsUploadId=None, artifactsUploadId=None, verificationsUploadId=None, defaultItemsUploadId=None, modelFileUploadId=None, modelStateUploadId=None, inputPreprocessorUploadId=None, requirementsUploadId=None, resourcesUploadId=None, multiCatalogEmbeddingsUploadId=None):
        super().__init__(client, modelUploadId)
        self.model_id = modelId
        self.model_version = modelVersion
        self.status = status
        self.created_at = createdAt
        self.model_upload_id = modelUploadId
        self.embeddings_upload_id = embeddingsUploadId
        self.artifacts_upload_id = artifactsUploadId
        self.verifications_upload_id = verificationsUploadId
        self.default_items_upload_id = defaultItemsUploadId
        self.model_file_upload_id = modelFileUploadId
        self.model_state_upload_id = modelStateUploadId
        self.input_preprocessor_upload_id = inputPreprocessorUploadId
        self.requirements_upload_id = requirementsUploadId
        self.resources_upload_id = resourcesUploadId
        self.multi_catalog_embeddings_upload_id = multiCatalogEmbeddingsUploadId

    def __repr__(self):
        repr_dict = {f'model_id': repr(self.model_id), f'model_version': repr(self.model_version), f'status': repr(self.status), f'created_at': repr(self.created_at), f'model_upload_id': repr(self.model_upload_id), f'embeddings_upload_id': repr(self.embeddings_upload_id), f'artifacts_upload_id': repr(self.artifacts_upload_id), f'verifications_upload_id': repr(self.verifications_upload_id), f'default_items_upload_id': repr(
            self.default_items_upload_id), f'model_file_upload_id': repr(self.model_file_upload_id), f'model_state_upload_id': repr(self.model_state_upload_id), f'input_preprocessor_upload_id': repr(self.input_preprocessor_upload_id), f'requirements_upload_id': repr(self.requirements_upload_id), f'resources_upload_id': repr(self.resources_upload_id), f'multi_catalog_embeddings_upload_id': repr(self.multi_catalog_embeddings_upload_id)}
        class_name = "ModelUpload"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model_id': self.model_id, 'model_version': self.model_version, 'status': self.status, 'created_at': self.created_at, 'model_upload_id': self.model_upload_id, 'embeddings_upload_id': self.embeddings_upload_id, 'artifacts_upload_id': self.artifacts_upload_id, 'verifications_upload_id': self.verifications_upload_id, 'default_items_upload_id': self.default_items_upload_id,
                'model_file_upload_id': self.model_file_upload_id, 'model_state_upload_id': self.model_state_upload_id, 'input_preprocessor_upload_id': self.input_preprocessor_upload_id, 'requirements_upload_id': self.requirements_upload_id, 'resources_upload_id': self.resources_upload_id, 'multi_catalog_embeddings_upload_id': self.multi_catalog_embeddings_upload_id}
        return {key: value for key, value in resp.items() if value is not None}
