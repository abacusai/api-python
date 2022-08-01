from .drift_distribution import DriftDistribution
from .return_class import AbstractApiClass


class DriftDistributions(AbstractApiClass):
    """
        For either actuals or predicted values, how it has changed in the training data versus some specified window

        Args:
            client (ApiClient): An authenticated API Client instance
            labelDrift (DriftDistribution): A DriftDistribution describing column names and the range of values for label drift.
            predictionDrift (DriftDistribution): A DriftDistribution describing column names and the range of values for prediction drift.
    """

    def __init__(self, client, labelDrift={}, predictionDrift={}):
        super().__init__(client, None)
        self.label_drift = client._build_class(DriftDistribution, labelDrift)
        self.prediction_drift = client._build_class(
            DriftDistribution, predictionDrift)

    def __repr__(self):
        return f"DriftDistributions(label_drift={repr(self.label_drift)},\n  prediction_drift={repr(self.prediction_drift)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'label_drift': self._get_attribute_as_dict(self.label_drift), 'prediction_drift': self._get_attribute_as_dict(self.prediction_drift)}
