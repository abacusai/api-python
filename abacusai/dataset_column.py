

class DatasetColumn():
    '''
        A schema description for a column
    '''

    def __init__(self, client, name=None, dataType=None, featureType=None, originalName=None):
        self.client = client
        self.id = None
        self.name = name
        self.data_type = dataType
        self.feature_type = featureType
        self.original_name = originalName

    def __repr__(self):
        return f"DatasetColumn(name={repr(self.name)}, data_type={repr(self.data_type)}, feature_type={repr(self.feature_type)}, original_name={repr(self.original_name)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'data_type': self.data_type, 'feature_type': self.feature_type, 'original_name': self.original_name}
