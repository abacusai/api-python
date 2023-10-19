from .return_class import AbstractApiClass


class TestPointPredictions(AbstractApiClass):
    """
        Test Point Predictions

        Args:
            client (ApiClient): An authenticated API Client instance
            count (int): Count of total rows in the preview data for the SQL.
            columns (list): The returned columns
            data (list): A list of data rows, each represented as a list.
            summarizedMetrics (dict): A map between the problem type metrics and the mean of the results matching the query
            errorDescription (str): Description of an error in case of failure.
    """

    def __init__(self, client, count=None, columns=None, data=None, summarizedMetrics=None, errorDescription=None):
        super().__init__(client, None)
        self.count = count
        self.columns = columns
        self.data = data
        self.summarized_metrics = summarizedMetrics
        self.error_description = errorDescription

    def __repr__(self):
        repr_dict = {f'count': repr(self.count), f'columns': repr(self.columns), f'data': repr(
            self.data), f'summarized_metrics': repr(self.summarized_metrics), f'error_description': repr(self.error_description)}
        class_name = "TestPointPredictions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'count': self.count, 'columns': self.columns, 'data': self.data,
                'summarized_metrics': self.summarized_metrics, 'error_description': self.error_description}
        return {key: value for key, value in resp.items() if value is not None}
