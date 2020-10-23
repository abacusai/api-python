

class ModelUpload():
    '''

    '''

    def __init__(self, client, modelId=None, modelVersion=None, status=None, createdAt=None, modelUploadId=None, embeddingsUploadId=None, verificationsUploadId=None, defaultItemsUploadId=None):
        self.client = client
        self.id = modelUploadId
        self.model_id = modelId
        self.model_version = modelVersion
        self.status = status
        self.created_at = createdAt
        self.model_upload_id = modelUploadId
        self.embeddings_upload_id = embeddingsUploadId
        self.verifications_upload_id = verificationsUploadId
        self.default_items_upload_id = defaultItemsUploadId

    def __repr__(self):
        return f"ModelUpload(model_id={repr(self.model_id)}, model_version={repr(self.model_version)}, status={repr(self.status)}, created_at={repr(self.created_at)}, model_upload_id={repr(self.model_upload_id)}, embeddings_upload_id={repr(self.embeddings_upload_id)}, verifications_upload_id={repr(self.verifications_upload_id)}, default_items_upload_id={repr(self.default_items_upload_id)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'model_id': self.model_id, 'model_version': self.model_version, 'status': self.status, 'created_at': self.created_at, 'model_upload_id': self.model_upload_id, 'embeddings_upload_id': self.embeddings_upload_id, 'verifications_upload_id': self.verifications_upload_id, 'default_items_upload_id': self.default_items_upload_id}
