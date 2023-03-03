from .forecasting_analysis_graph_data import ForecastingAnalysisGraphData
from .return_class import AbstractApiClass


class EdaForecastingAnalysis(AbstractApiClass):
    """
        Eda Forecasting Analysis of the latest version of the data.

        Args:
            client (ApiClient): An authenticated API Client instance
            primaryKeys (list): Name of the primary keys in the data
            forecastingTargetFeature (str): Feature in the data that represents the target.
            timestampFeature (str): Feature in the data that represents the timestamp column.
            forecastFrequency (str): Frequency of data, could be hourly, daily, weekly, monthly, quarterly or yearly.
            salesAcrossTime (ForecastingAnalysisGraphData): Data showing average, p10, p90, median sales across time
            cummulativeContribution (ForecastingAnalysisGraphData): Data showing what percent of items contribute to what amount of sales.
            missingValueDistribution (ForecastingAnalysisGraphData): Data showing missing or null value distribution
            historyLength (ForecastingAnalysisGraphData): Data showing length of history distribution
            numRowsHistogram (ForecastingAnalysisGraphData): Data showing number of rows for an item distribution
            productMaturity (ForecastingAnalysisGraphData): Data showing length of how long a product has been alive with average, p10, p90 and median
    """

    def __init__(self, client, primaryKeys=None, forecastingTargetFeature=None, timestampFeature=None, forecastFrequency=None, salesAcrossTime={}, cummulativeContribution={}, missingValueDistribution={}, historyLength={}, numRowsHistogram={}, productMaturity={}):
        super().__init__(client, None)
        self.primary_keys = primaryKeys
        self.forecasting_target_feature = forecastingTargetFeature
        self.timestamp_feature = timestampFeature
        self.forecast_frequency = forecastFrequency
        self.sales_across_time = client._build_class(
            ForecastingAnalysisGraphData, salesAcrossTime)
        self.cummulative_contribution = client._build_class(
            ForecastingAnalysisGraphData, cummulativeContribution)
        self.missing_value_distribution = client._build_class(
            ForecastingAnalysisGraphData, missingValueDistribution)
        self.history_length = client._build_class(
            ForecastingAnalysisGraphData, historyLength)
        self.num_rows_histogram = client._build_class(
            ForecastingAnalysisGraphData, numRowsHistogram)
        self.product_maturity = client._build_class(
            ForecastingAnalysisGraphData, productMaturity)

    def __repr__(self):
        return f"EdaForecastingAnalysis(primary_keys={repr(self.primary_keys)},\n  forecasting_target_feature={repr(self.forecasting_target_feature)},\n  timestamp_feature={repr(self.timestamp_feature)},\n  forecast_frequency={repr(self.forecast_frequency)},\n  sales_across_time={repr(self.sales_across_time)},\n  cummulative_contribution={repr(self.cummulative_contribution)},\n  missing_value_distribution={repr(self.missing_value_distribution)},\n  history_length={repr(self.history_length)},\n  num_rows_histogram={repr(self.num_rows_histogram)},\n  product_maturity={repr(self.product_maturity)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'primary_keys': self.primary_keys, 'forecasting_target_feature': self.forecasting_target_feature, 'timestamp_feature': self.timestamp_feature, 'forecast_frequency': self.forecast_frequency, 'sales_across_time': self._get_attribute_as_dict(self.sales_across_time), 'cummulative_contribution': self._get_attribute_as_dict(self.cummulative_contribution), 'missing_value_distribution': self._get_attribute_as_dict(self.missing_value_distribution), 'history_length': self._get_attribute_as_dict(self.history_length), 'num_rows_histogram': self._get_attribute_as_dict(self.num_rows_histogram), 'product_maturity': self._get_attribute_as_dict(self.product_maturity)}
