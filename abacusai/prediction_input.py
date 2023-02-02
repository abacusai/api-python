from .prediction_dataset import PredictionDataset
from .prediction_feature_group import PredictionFeatureGroup
from .return_class import AbstractApiClass


class PredictionInput(AbstractApiClass):
    """
        Batch inputs

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupDatasetIds (list): The list of dataset IDs to use as input
            datasetIdRemap (dict): Replacement datasets to swap as prediction input
            featureGroups (PredictionFeatureGroup): List of prediction feature groups
            datasets (PredictionDataset): List of prediction datasets
    """

    def __init__(self, client, featureGroupDatasetIds=None, datasetIdRemap=None, featureGroups={}, datasets={}):
        super().__init__(client, None)
        self.feature_group_dataset_ids = featureGroupDatasetIds
        self.dataset_id_remap = datasetIdRemap
        self.feature_groups = client._build_class(
            PredictionFeatureGroup, featureGroups)
        self.datasets = client._build_class(PredictionDataset, datasets)

    def __repr__(self):
        return f"PredictionInput(feature_group_dataset_ids={repr(self.feature_group_dataset_ids)},\n  dataset_id_remap={repr(self.dataset_id_remap)},\n  feature_groups={repr(self.feature_groups)},\n  datasets={repr(self.datasets)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_dataset_ids': self.feature_group_dataset_ids, 'dataset_id_remap': self.dataset_id_remap, 'feature_groups': self._get_attribute_as_dict(self.feature_groups), 'datasets': self._get_attribute_as_dict(self.datasets)}
