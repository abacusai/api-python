from .return_class import AbstractApiClass


class PredictionFeatureGroup(AbstractApiClass):
    """
        Batch Input Feature Group
    """

    def __init__(self, client, featureGroupId=None, datasetType=None, default=None, required=None):
        super().__init__(client, None)
        self.feature_group_id = featureGroupId
        self.dataset_type = datasetType
        self.default = default
        self.required = required

    def __repr__(self):
        return f"PredictionFeatureGroup(feature_group_id={repr(self.feature_group_id)},\n  dataset_type={repr(self.dataset_type)},\n  default={repr(self.default)},\n  required={repr(self.required)})"

    def to_dict(self):
        return {'feature_group_id': self.feature_group_id, 'dataset_type': self.dataset_type, 'default': self.default, 'required': self.required}
