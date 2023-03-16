from .return_class import AbstractApiClass


class EdaChartDescription(AbstractApiClass):
    """
        Eda Chart Description.

        Args:
            client (ApiClient): An authenticated API Client instance
            chartType (str): Name of chart.
            description (str): Description of the eda chart.
    """

    def __init__(self, client, chartType=None, description=None):
        super().__init__(client, None)
        self.chart_type = chartType
        self.description = description

    def __repr__(self):
        return f"EdaChartDescription(chart_type={repr(self.chart_type)},\n  description={repr(self.description)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'chart_type': self.chart_type, 'description': self.description}
