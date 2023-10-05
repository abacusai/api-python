from .forecasting_analysis_graph_data import ForecastingAnalysisGraphData
from .return_class import AbstractApiClass


class ForecastingMonitorSummary(AbstractApiClass):
    """
        Forecasting Monitor Summary of the latest version of the data.

        Args:
            client (ApiClient): An authenticated API Client instance
            predictionTimestampCol (str): Feature in the data that represents the timestamp column.
            predictionTargetCol (str): Feature in the data that represents the target.
            trainingTimestampCol (str): Feature in the data that represents the timestamp column.
            trainingTargetCol (str): Feature in the data that represents the target.
            predictionItemId (str): Feature in the data that represents the item id.
            trainingItemId (str): Feature in the data that represents the item id.
            forecastFrequency (str): Frequency of data, could be hourly, daily, weekly, monthly, quarterly or yearly.
            trainingTargetAcrossTime (ForecastingAnalysisGraphData): Data showing average, p10, p90, median sales across time
            predictionTargetAcrossTime (ForecastingAnalysisGraphData): Data showing average, p10, p90, median sales across time
            actualsHistogram (ForecastingAnalysisGraphData): Data showing actuals histogram
            predictionsHistogram (ForecastingAnalysisGraphData): Data showing predictions histogram
            trainHistoryData (ForecastingAnalysisGraphData): Data showing length of history distribution
            predictHistoryData (ForecastingAnalysisGraphData): Data showing length of history distribution
    """

    def __init__(self, client, predictionTimestampCol=None, predictionTargetCol=None, trainingTimestampCol=None, trainingTargetCol=None, predictionItemId=None, trainingItemId=None, forecastFrequency=None, trainingTargetAcrossTime={}, predictionTargetAcrossTime={}, actualsHistogram={}, predictionsHistogram={}, trainHistoryData={}, predictHistoryData={}):
        super().__init__(client, None)
        self.prediction_timestamp_col = predictionTimestampCol
        self.prediction_target_col = predictionTargetCol
        self.training_timestamp_col = trainingTimestampCol
        self.training_target_col = trainingTargetCol
        self.prediction_item_id = predictionItemId
        self.training_item_id = trainingItemId
        self.forecast_frequency = forecastFrequency
        self.training_target_across_time = client._build_class(
            ForecastingAnalysisGraphData, trainingTargetAcrossTime)
        self.prediction_target_across_time = client._build_class(
            ForecastingAnalysisGraphData, predictionTargetAcrossTime)
        self.actuals_histogram = client._build_class(
            ForecastingAnalysisGraphData, actualsHistogram)
        self.predictions_histogram = client._build_class(
            ForecastingAnalysisGraphData, predictionsHistogram)
        self.train_history_data = client._build_class(
            ForecastingAnalysisGraphData, trainHistoryData)
        self.predict_history_data = client._build_class(
            ForecastingAnalysisGraphData, predictHistoryData)

    def __repr__(self):
        return f"ForecastingMonitorSummary(prediction_timestamp_col={repr(self.prediction_timestamp_col)},\n  prediction_target_col={repr(self.prediction_target_col)},\n  training_timestamp_col={repr(self.training_timestamp_col)},\n  training_target_col={repr(self.training_target_col)},\n  prediction_item_id={repr(self.prediction_item_id)},\n  training_item_id={repr(self.training_item_id)},\n  forecast_frequency={repr(self.forecast_frequency)},\n  training_target_across_time={repr(self.training_target_across_time)},\n  prediction_target_across_time={repr(self.prediction_target_across_time)},\n  actuals_histogram={repr(self.actuals_histogram)},\n  predictions_histogram={repr(self.predictions_histogram)},\n  train_history_data={repr(self.train_history_data)},\n  predict_history_data={repr(self.predict_history_data)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'prediction_timestamp_col': self.prediction_timestamp_col, 'prediction_target_col': self.prediction_target_col, 'training_timestamp_col': self.training_timestamp_col, 'training_target_col': self.training_target_col, 'prediction_item_id': self.prediction_item_id, 'training_item_id': self.training_item_id, 'forecast_frequency': self.forecast_frequency, 'training_target_across_time': self._get_attribute_as_dict(self.training_target_across_time), 'prediction_target_across_time': self._get_attribute_as_dict(self.prediction_target_across_time), 'actuals_histogram': self._get_attribute_as_dict(self.actuals_histogram), 'predictions_histogram': self._get_attribute_as_dict(self.predictions_histogram), 'train_history_data': self._get_attribute_as_dict(self.train_history_data), 'predict_history_data': self._get_attribute_as_dict(self.predict_history_data)}
