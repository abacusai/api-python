from .data_filter import DataFilter


class ProjectDataset():
    '''
        The description of how a dataset is used in a project
    '''

    def __init__(self, client, name=None, datasetType=None, datasetId=None, streaming=None, dataFilters={}):
        self.client = client
        self.id = None
        self.name = name
        self.dataset_type = datasetType
        self.dataset_id = datasetId
        self.streaming = streaming
        self.data_filters = client._build_class(DataFilter, dataFilters)

    def __repr__(self):
        return f"ProjectDataset(name={repr(self.name)}, dataset_type={repr(self.dataset_type)}, dataset_id={repr(self.dataset_id)}, streaming={repr(self.streaming)}, data_filters={repr(self.data_filters)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'dataset_type': self.dataset_type, 'dataset_id': self.dataset_id, 'streaming': self.streaming, 'data_filters': self.data_filters.to_dict() if self.data_filters else None}
