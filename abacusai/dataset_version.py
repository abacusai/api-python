from .upload import Upload


class DatasetVersion():
    '''
        A specific version of a dataset
    '''

    def __init__(self, client, datasetVersion=None, status=None, datasetId=None, size=None, createdAt=None, error=None):
        self.client = client
        self.id = datasetVersion
        self.dataset_version = datasetVersion
        self.status = status
        self.dataset_id = datasetId
        self.size = size
        self.created_at = createdAt
        self.error = error

    def __repr__(self):
        return f"DatasetVersion(dataset_version={repr(self.dataset_version)}, status={repr(self.status)}, dataset_id={repr(self.dataset_id)}, size={repr(self.size)}, created_at={repr(self.created_at)}, error={repr(self.error)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'dataset_version': self.dataset_version, 'status': self.status, 'dataset_id': self.dataset_id, 'size': self.size, 'created_at': self.created_at, 'error': self.error}

    def wait_for_import(self, timeout=900):
        return self.client._poll(self, {'PENDING', 'IMPORTING'}, timeout=timeout)

    def wait_for_inspection(self, timeout=None):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'IMPORTING', 'CONVERTING', 'INSPECTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status
