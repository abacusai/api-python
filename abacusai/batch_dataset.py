

class BatchDataset():
    '''

    '''

    def __init__(self, client, datasetId=None, status=None, uploaded=None, datasetType=None):
        self.client = client
        self.id = None
        self.dataset_id = datasetId
        self.status = status
        self.uploaded = uploaded
        self.dataset_type = datasetType

    def __repr__(self):
        return f"BatchDataset(dataset_id={repr(self.dataset_id)}, status={repr(self.status)}, uploaded={repr(self.uploaded)}, dataset_type={repr(self.dataset_type)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'dataset_id': self.dataset_id, 'status': self.status, 'uploaded': self.uploaded, 'dataset_type': self.dataset_type}
