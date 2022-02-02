from .return_class import AbstractApiClass


class ModelUpload(AbstractApiClass):
    """
        A Model Version that includes the upload ids for the various required files.

        Args:
            client (ApiClient): An authenticated API Client instance
            modelId (str): A reference to the model this version belongs to
            modelVersion (str): The unique identifier of a model version.
            status (str): The current status of the model.
            createdAt (str): The timestamp at which the model version was created.
            modelUploadId (str): An Upload Identifier to be used when uploading the Tensorflow Saved Model
            embeddingsUploadId (str): An Upload Identifier to be used when uploading the embeddings csv
            artifactsUploadId (str): An Upload Identifier to be used when uploading the artifacts json file
            verificationsUploadId (str): An Upload Identifier to be used when uploading the verifications json file
            defaultItemsUploadId (str): An Upload Identifier to be used when uploading the default items json file
            modelFileUploadId (str): An Upload Identifier to be used when uploading the model json file
            modelStateUploadId (str): An Upload Identifier to be used when uploading the model state json file
            inputPreprocessorUploadId (str): An Upload Identifier to be used when uploading the input preprocessor json file
            requirementsUploadId (str): An Upload Identifier to be used when uploading the requirements json file
            resourcesUploadId (str): An Upload Identifier to be used when uploading the resources json file
    """

    def __init__(self, client, modelId=None, modelVersion=None, status=None, createdAt=None, modelUploadId=None, embeddingsUploadId=None, artifactsUploadId=None, verificationsUploadId=None, defaultItemsUploadId=None, modelFileUploadId=None, modelStateUploadId=None, inputPreprocessorUploadId=None, requirementsUploadId=None, resourcesUploadId=None):
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

    def __repr__(self):
        return f"ModelUpload(model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  status={repr(self.status)},\n  created_at={repr(self.created_at)},\n  model_upload_id={repr(self.model_upload_id)},\n  embeddings_upload_id={repr(self.embeddings_upload_id)},\n  artifacts_upload_id={repr(self.artifacts_upload_id)},\n  verifications_upload_id={repr(self.verifications_upload_id)},\n  default_items_upload_id={repr(self.default_items_upload_id)},\n  model_file_upload_id={repr(self.model_file_upload_id)},\n  model_state_upload_id={repr(self.model_state_upload_id)},\n  input_preprocessor_upload_id={repr(self.input_preprocessor_upload_id)},\n  requirements_upload_id={repr(self.requirements_upload_id)},\n  resources_upload_id={repr(self.resources_upload_id)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_id': self.model_id, 'model_version': self.model_version, 'status': self.status, 'created_at': self.created_at, 'model_upload_id': self.model_upload_id, 'embeddings_upload_id': self.embeddings_upload_id, 'artifacts_upload_id': self.artifacts_upload_id, 'verifications_upload_id': self.verifications_upload_id, 'default_items_upload_id': self.default_items_upload_id, 'model_file_upload_id': self.model_file_upload_id, 'model_state_upload_id': self.model_state_upload_id, 'input_preprocessor_upload_id': self.input_preprocessor_upload_id, 'requirements_upload_id': self.requirements_upload_id, 'resources_upload_id': self.resources_upload_id}
