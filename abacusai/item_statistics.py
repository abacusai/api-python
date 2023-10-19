from .return_class import AbstractApiClass


class ItemStatistics(AbstractApiClass):
    """
        ItemStatistics representation.

        Args:
            client (ApiClient): An authenticated API Client instance
            missingPercent (float): percentage of missing values in data
            count (int): count of data
            median (float): median of the data
            mean (float): mean value of the data
            p10 (float): 10th percentile of the data
            p90 (float): 90th_percentile of the data
            stddev (float): standard deviation of the data
            min (int): min value in the data
            max (int): max value in the data
    """

    def __init__(self, client, missingPercent=None, count=None, median=None, mean=None, p10=None, p90=None, stddev=None, min=None, max=None):
        super().__init__(client, None)
        self.missing_percent = missingPercent
        self.count = count
        self.median = median
        self.mean = mean
        self.p10 = p10
        self.p90 = p90
        self.stddev = stddev
        self.min = min
        self.max = max

    def __repr__(self):
        repr_dict = {f'missing_percent': repr(self.missing_percent), f'count': repr(self.count), f'median': repr(self.median), f'mean': repr(
            self.mean), f'p10': repr(self.p10), f'p90': repr(self.p90), f'stddev': repr(self.stddev), f'min': repr(self.min), f'max': repr(self.max)}
        class_name = "ItemStatistics"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'missing_percent': self.missing_percent, 'count': self.count, 'median': self.median,
                'mean': self.mean, 'p10': self.p10, 'p90': self.p90, 'stddev': self.stddev, 'min': self.min, 'max': self.max}
        return {key: value for key, value in resp.items() if value is not None}
