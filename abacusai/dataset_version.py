from .return_class import AbstractApiClass
from .upload import Upload


class DatasetVersion(AbstractApiClass):
    """
        A specific version of a dataset
    """

    def __init__(self, client, datasetVersion=None, status=None, datasetId=None, size=None, rowCount=None, createdAt=None, error=None):
        super().__init__(client, datasetVersion)
        self.dataset_version = datasetVersion
        self.status = status
        self.dataset_id = datasetId
        self.size = size
        self.row_count = rowCount
        self.created_at = createdAt
        self.error = error

    def __repr__(self):
        return f"DatasetVersion(dataset_version={repr(self.dataset_version)},\n  status={repr(self.status)},\n  dataset_id={repr(self.dataset_id)},\n  size={repr(self.size)},\n  row_count={repr(self.row_count)},\n  created_at={repr(self.created_at)},\n  error={repr(self.error)})"

    def to_dict(self):
        return {'dataset_version': self.dataset_version, 'status': self.status, 'dataset_id': self.dataset_id, 'size': self.size, 'row_count': self.row_count, 'created_at': self.created_at, 'error': self.error}

    def wait_for_import(self, timeout=900):
        return self.client._poll(self, {'PENDING', 'IMPORTING'}, timeout=timeout)

    def wait_for_inspection(self, timeout=None):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'IMPORTING', 'CONVERTING', 'INSPECTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status
