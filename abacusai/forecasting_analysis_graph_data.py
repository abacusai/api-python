from .eda_chart_description import EdaChartDescription
from .item_statistics import ItemStatistics
from .return_class import AbstractApiClass


class ForecastingAnalysisGraphData(AbstractApiClass):
    """
        Forecasting Analysis Graph Data representation.

        Args:
            client (ApiClient): An authenticated API Client instance
            data (list): List of graph data
            xAxis (str): Feature that represents the x axis
            yAxis (str): Feature that represents the y axis
            dataColumns (list): Ordered name of the column for each rowwise data
            chartName (str): Name of the chart represented by the data
            chartTypes (list): Type of charts in that can exist in the current data.
            itemStatistics (ItemStatistics): In item wise charts, gives the mean, median, count, missing_percent, p10, p90, standard_deviation, min, max
            chartDescriptions (EdaChartDescription): List of descriptions of what the chart contains
    """

    def __init__(self, client, data=None, xAxis=None, yAxis=None, dataColumns=None, chartName=None, chartTypes=None, itemStatistics={}, chartDescriptions={}):
        super().__init__(client, None)
        self.data = data
        self.x_axis = xAxis
        self.y_axis = yAxis
        self.data_columns = dataColumns
        self.chart_name = chartName
        self.chart_types = chartTypes
        self.item_statistics = client._build_class(
            ItemStatistics, itemStatistics)
        self.chart_descriptions = client._build_class(
            EdaChartDescription, chartDescriptions)

    def __repr__(self):
        repr_dict = {f'data': repr(self.data), f'x_axis': repr(self.x_axis), f'y_axis': repr(self.y_axis), f'data_columns': repr(self.data_columns), f'chart_name': repr(
            self.chart_name), f'chart_types': repr(self.chart_types), f'item_statistics': repr(self.item_statistics), f'chart_descriptions': repr(self.chart_descriptions)}
        class_name = "ForecastingAnalysisGraphData"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'data': self.data, 'x_axis': self.x_axis, 'y_axis': self.y_axis, 'data_columns': self.data_columns, 'chart_name': self.chart_name, 'chart_types': self.chart_types,
                'item_statistics': self._get_attribute_as_dict(self.item_statistics), 'chart_descriptions': self._get_attribute_as_dict(self.chart_descriptions)}
        return {key: value for key, value in resp.items() if value is not None}
