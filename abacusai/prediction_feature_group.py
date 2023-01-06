from .return_class import AbstractApiClass


class PredictionFeatureGroup(AbstractApiClass):
    """
        Batch Input Feature Group

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): The unique identifier of the feature group
            featureGroupVersion (str): The unique identifier of the feature group version used for predictions
            datasetType (str): dataset type
            default (bool): If true, this feature group is the default feature group in the model
            required (bool): If true, this feature group is required for the batch prediction
    """

    def __init__(self, client, featureGroupId=None, featureGroupVersion=None, datasetType=None, default=None, required=None):
        super().__init__(client, None)
        self.feature_group_id = featureGroupId
        self.feature_group_version = featureGroupVersion
        self.dataset_type = datasetType
        self.default = default
        self.required = required

    def __repr__(self):
        return f"PredictionFeatureGroup(feature_group_id={repr(self.feature_group_id)},\n  feature_group_version={repr(self.feature_group_version)},\n  dataset_type={repr(self.dataset_type)},\n  default={repr(self.default)},\n  required={repr(self.required)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_id': self.feature_group_id, 'feature_group_version': self.feature_group_version, 'dataset_type': self.dataset_type, 'default': self.default, 'required': self.required}
