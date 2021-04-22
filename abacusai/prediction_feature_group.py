

class PredictionFeatureGroup():
    '''

    '''

    def __init__(self, client, featureGroupId=None, datasetType=None, default=None, required=None):
        self.client = client
        self.id = None
        self.feature_group_id = featureGroupId
        self.dataset_type = datasetType
        self.default = default
        self.required = required

    def __repr__(self):
        return f"PredictionFeatureGroup(feature_group_id={repr(self.feature_group_id)}, dataset_type={repr(self.dataset_type)}, default={repr(self.default)}, required={repr(self.required)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_id': self.feature_group_id, 'dataset_type': self.dataset_type, 'default': self.default, 'required': self.required}
