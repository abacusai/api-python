from .return_class import AbstractApiClass


class PredictionDataset(AbstractApiClass):
    """
        Batch Input Datasets

        Args:
            client (ApiClient): An authenticated API Client instance
            datasetId (str): The unique identifier of the dataset
            datasetType (str): dataset type
            datasetVersion (str): The unique identifier of the dataset version used for predictions
            default (bool): If true, this dataset is the default dataset in the model
            required (bool): If true...
    """

    def __init__(self, client, datasetId=None, datasetType=None, datasetVersion=None, default=None, required=None):
        super().__init__(client, None)
        self.dataset_id = datasetId
        self.dataset_type = datasetType
        self.dataset_version = datasetVersion
        self.default = default
        self.required = required

    def __repr__(self):
        return f"PredictionDataset(dataset_id={repr(self.dataset_id)},\n  dataset_type={repr(self.dataset_type)},\n  dataset_version={repr(self.dataset_version)},\n  default={repr(self.default)},\n  required={repr(self.required)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'dataset_id': self.dataset_id, 'dataset_type': self.dataset_type, 'dataset_version': self.dataset_version, 'default': self.default, 'required': self.required}
