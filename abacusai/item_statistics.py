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
        return f"ItemStatistics(missing_percent={repr(self.missing_percent)},\n  count={repr(self.count)},\n  median={repr(self.median)},\n  mean={repr(self.mean)},\n  p10={repr(self.p10)},\n  p90={repr(self.p90)},\n  stddev={repr(self.stddev)},\n  min={repr(self.min)},\n  max={repr(self.max)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'missing_percent': self.missing_percent, 'count': self.count, 'median': self.median, 'mean': self.mean, 'p10': self.p10, 'p90': self.p90, 'stddev': self.stddev, 'min': self.min, 'max': self.max}
