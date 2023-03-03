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
            itemStatistics (ItemStatistics): In item wise charts, gives the mean, median, count, missing_percent, p10, p90, standard_deviation, min, max
    """

    def __init__(self, client, data=None, xAxis=None, yAxis=None, dataColumns=None, itemStatistics={}):
        super().__init__(client, None)
        self.data = data
        self.x_axis = xAxis
        self.y_axis = yAxis
        self.data_columns = dataColumns
        self.item_statistics = client._build_class(
            ItemStatistics, itemStatistics)

    def __repr__(self):
        return f"ForecastingAnalysisGraphData(data={repr(self.data)},\n  x_axis={repr(self.x_axis)},\n  y_axis={repr(self.y_axis)},\n  data_columns={repr(self.data_columns)},\n  item_statistics={repr(self.item_statistics)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'data': self.data, 'x_axis': self.x_axis, 'y_axis': self.y_axis, 'data_columns': self.data_columns, 'item_statistics': self._get_attribute_as_dict(self.item_statistics)}
