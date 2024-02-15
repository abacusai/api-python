from .forecasting_analysis_graph_data import ForecastingAnalysisGraphData
from .return_class import AbstractApiClass


class ForecastingMonitorItemAnalysis(AbstractApiClass):
    """
        Forecasting Monitor Item Analysis of the latest version of the data.

        Args:
            client (ApiClient): An authenticated API Client instance
            predictionItemAnalysis (ForecastingAnalysisGraphData): Data showing average, p10, p90, median sales across time for prediction data
            trainingItemAnalysis (ForecastingAnalysisGraphData): Data showing average, p10, p90, median sales across time for training data
    """

    def __init__(self, client, predictionItemAnalysis={}, trainingItemAnalysis={}):
        super().__init__(client, None)
        self.prediction_item_analysis = client._build_class(
            ForecastingAnalysisGraphData, predictionItemAnalysis)
        self.training_item_analysis = client._build_class(
            ForecastingAnalysisGraphData, trainingItemAnalysis)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'prediction_item_analysis': repr(
            self.prediction_item_analysis), f'training_item_analysis': repr(self.training_item_analysis)}
        class_name = "ForecastingMonitorItemAnalysis"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'prediction_item_analysis': self._get_attribute_as_dict(
            self.prediction_item_analysis), 'training_item_analysis': self._get_attribute_as_dict(self.training_item_analysis)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
