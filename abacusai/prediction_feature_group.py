from .return_class import AbstractApiClass


class PredictionFeatureGroup(AbstractApiClass):
    """
        Batch Input Feature Group

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): The unique identifier of the dataset
            datasetType (str): dataset type
            default (bool): If true, this dataset is the default feature group in the model
            required (bool): If true...
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
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_id': self.feature_group_id, 'dataset_type': self.dataset_type, 'default': self.default, 'required': self.required}
