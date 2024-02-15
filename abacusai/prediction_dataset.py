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
            required (bool): If true, this dataset is required for the batch prediction
    """

    def __init__(self, client, datasetId=None, datasetType=None, datasetVersion=None, default=None, required=None):
        super().__init__(client, None)
        self.dataset_id = datasetId
        self.dataset_type = datasetType
        self.dataset_version = datasetVersion
        self.default = default
        self.required = required
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'dataset_id': repr(self.dataset_id), f'dataset_type': repr(self.dataset_type), f'dataset_version': repr(
            self.dataset_version), f'default': repr(self.default), f'required': repr(self.required)}
        class_name = "PredictionDataset"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'dataset_id': self.dataset_id, 'dataset_type': self.dataset_type,
                'dataset_version': self.dataset_version, 'default': self.default, 'required': self.required}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
