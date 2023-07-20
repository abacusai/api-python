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
        return f"TestPointPredictions(count={repr(self.count)},\n  columns={repr(self.columns)},\n  data={repr(self.data)},\n  summarized_metrics={repr(self.summarized_metrics)},\n  error_description={repr(self.error_description)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'count': self.count, 'columns': self.columns, 'data': self.data, 'summarized_metrics': self.summarized_metrics, 'error_description': self.error_description}
