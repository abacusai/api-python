

class PredictionDataset():
    '''

    '''

    def __init__(self, client, datasetId=None, datasetType=None, datasetVersion=None, default=None, required=None):
        self.client = client
        self.id = None
        self.dataset_id = datasetId
        self.dataset_type = datasetType
        self.dataset_version = datasetVersion
        self.default = default
        self.required = required

    def __repr__(self):
        return f"PredictionDataset(dataset_id={repr(self.dataset_id)}, dataset_type={repr(self.dataset_type)}, dataset_version={repr(self.dataset_version)}, default={repr(self.default)}, required={repr(self.required)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'dataset_id': self.dataset_id, 'dataset_type': self.dataset_type, 'dataset_version': self.dataset_version, 'default': self.default, 'required': self.required}
