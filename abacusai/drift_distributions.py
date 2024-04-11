from .drift_distribution import DriftDistribution
from .return_class import AbstractApiClass


class DriftDistributions(AbstractApiClass):
    """
        For either actuals or predicted values, how it has changed in the training data versus some specified window

        Args:
            client (ApiClient): An authenticated API Client instance
            labelDrift (DriftDistribution): A DriftDistribution describing column names and the range of values for label drift.
            predictionDrift (DriftDistribution): A DriftDistribution describing column names and the range of values for prediction drift.
            bpPredictionDrift (DriftDistribution): A DriftDistribution describing column names and the range of values for prediction drift, when the predictions come from BP.
    """

    def __init__(self, client, labelDrift={}, predictionDrift={}, bpPredictionDrift={}):
        super().__init__(client, None)
        self.label_drift = client._build_class(DriftDistribution, labelDrift)
        self.prediction_drift = client._build_class(
            DriftDistribution, predictionDrift)
        self.bp_prediction_drift = client._build_class(
            DriftDistribution, bpPredictionDrift)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'label_drift': repr(self.label_drift), f'prediction_drift': repr(
            self.prediction_drift), f'bp_prediction_drift': repr(self.bp_prediction_drift)}
        class_name = "DriftDistributions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'label_drift': self._get_attribute_as_dict(self.label_drift), 'prediction_drift': self._get_attribute_as_dict(
            self.prediction_drift), 'bp_prediction_drift': self._get_attribute_as_dict(self.bp_prediction_drift)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
