from .return_class import AbstractApiClass


class DatasetColumn(AbstractApiClass):
    """
        A schema description for a column
    """

    def __init__(self, client, name=None, dataType=None, featureType=None, originalName=None):
        super().__init__(client, None)
        self.name = name
        self.data_type = dataType
        self.feature_type = featureType
        self.original_name = originalName

    def __repr__(self):
        return f"DatasetColumn(name={repr(self.name)},\n  data_type={repr(self.data_type)},\n  feature_type={repr(self.feature_type)},\n  original_name={repr(self.original_name)})"

    def to_dict(self):
        return {'name': self.name, 'data_type': self.data_type, 'feature_type': self.feature_type, 'original_name': self.original_name}
