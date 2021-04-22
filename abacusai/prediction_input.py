from .prediction_dataset import PredictionDataset
from .prediction_feature_group import PredictionFeatureGroup


class PredictionInput():
    '''

    '''

    def __init__(self, client, featureGroups={}, datasets={}):
        self.client = client
        self.id = None
        self.feature_groups = client._build_class(
            PredictionFeatureGroup, featureGroups)
        self.datasets = client._build_class(PredictionDataset, datasets)

    def __repr__(self):
        return f"PredictionInput(feature_groups={repr(self.feature_groups)}, datasets={repr(self.datasets)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_groups': [elem.to_dict() for elem in self.feature_groups or []], 'datasets': [elem.to_dict() for elem in self.datasets or []]}
