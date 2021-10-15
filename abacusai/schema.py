from .nested_feature import NestedFeature
from .point_in_time_feature import PointInTimeFeature


class Schema():
    '''
        A schema description for a feature
    '''

    def __init__(self, client, name=None, featureMapping=None, featureType=None, dataType=None):
        self.client = client
        self.id = None
        self.name = name
        self.feature_mapping = featureMapping
        self.feature_type = featureType
        self.data_type = dataType

    def __repr__(self):
        return f"Schema(name={repr(self.name)}, feature_mapping={repr(self.feature_mapping)}, feature_type={repr(self.feature_type)}, data_type={repr(self.data_type)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'feature_mapping': self.feature_mapping, 'feature_type': self.feature_type, 'data_type': self.data_type}
