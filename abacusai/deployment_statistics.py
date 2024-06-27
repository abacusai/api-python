from .return_class import AbstractApiClass


class DeploymentStatistics(AbstractApiClass):
    """
        A set of statistics for a realtime deployment.

        Args:
            client (ApiClient): An authenticated API Client instance
            requestSeries (list): A list of the number of requests per second.
            latencySeries (list): A list of the latency in milliseconds for each request.
            dateLabels (list): A list of date labels for each point in the series.
            httpStatusSeries (list): A list of the HTTP status codes for each request.
    """

    def __init__(self, client, requestSeries=None, latencySeries=None, dateLabels=None, httpStatusSeries=None):
        super().__init__(client, None)
        self.request_series = requestSeries
        self.latency_series = latencySeries
        self.date_labels = dateLabels
        self.http_status_series = httpStatusSeries
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'request_series': repr(self.request_series), f'latency_series': repr(
            self.latency_series), f'date_labels': repr(self.date_labels), f'http_status_series': repr(self.http_status_series)}
        class_name = "DeploymentStatistics"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'request_series': self.request_series, 'latency_series': self.latency_series,
                'date_labels': self.date_labels, 'http_status_series': self.http_status_series}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
