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
        repr_dict = {f'chart_type': repr(
            self.chart_type), f'description': repr(self.description)}
        class_name = "EdaChartDescription"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chart_type': self.chart_type, 'description': self.description}
        return {key: value for key, value in resp.items() if value is not None}
