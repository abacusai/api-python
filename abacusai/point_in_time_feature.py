from .return_class import AbstractApiClass


class PointInTimeFeature(AbstractApiClass):
    """
        A point-in-time feature description

        Args:
            client (ApiClient): An authenticated API Client instance
            historyTableName (str): The name of the history table. If not specified, the current table is used for a self-join.
            aggregationKeys (list[str]): List of keys to use for joining the historical table and performing the window aggregation.
            timestampKey (str): Name of feature which contains the timestamp value for the point-in-time feature.
            historicalTimestampKey (str): Name of feature which contains the historical timestamp.
            lookbackWindowSeconds (float): If window is specified in terms of time, the number of seconds in the past from the current time for the start of the window.
            lookbackWindowLagSeconds (float): Optional lag to offset the closest point for the window. If it is positive, the start of the window is delayed. If it is negative, we are looking at the "future" rows in the history table.
            lookbackCount (int): If window is specified in terms of count, the start position of the window (0 is the current row).
            lookbackUntilPosition (int): Optional lag to offset the closest point for the window. If it is positive, the start of the window is delayed by that many rows. If it is negative, we are looking at those many "future" rows in the history table.
            expression (str): SQL aggregate expression which can convert a sequence of rows into a scalar value.
            groupName (str): The group name this point-in-time feature belongs to.
    """

    def __init__(self, client, historyTableName=None, aggregationKeys=None, timestampKey=None, historicalTimestampKey=None, lookbackWindowSeconds=None, lookbackWindowLagSeconds=None, lookbackCount=None, lookbackUntilPosition=None, expression=None, groupName=None):
        super().__init__(client, None)
        self.history_table_name = historyTableName
        self.aggregation_keys = aggregationKeys
        self.timestamp_key = timestampKey
        self.historical_timestamp_key = historicalTimestampKey
        self.lookback_window_seconds = lookbackWindowSeconds
        self.lookback_window_lag_seconds = lookbackWindowLagSeconds
        self.lookback_count = lookbackCount
        self.lookback_until_position = lookbackUntilPosition
        self.expression = expression
        self.group_name = groupName
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'history_table_name': repr(self.history_table_name), f'aggregation_keys': repr(self.aggregation_keys), f'timestamp_key': repr(self.timestamp_key), f'historical_timestamp_key': repr(self.historical_timestamp_key), f'lookback_window_seconds': repr(
            self.lookback_window_seconds), f'lookback_window_lag_seconds': repr(self.lookback_window_lag_seconds), f'lookback_count': repr(self.lookback_count), f'lookback_until_position': repr(self.lookback_until_position), f'expression': repr(self.expression), f'group_name': repr(self.group_name)}
        class_name = "PointInTimeFeature"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'history_table_name': self.history_table_name, 'aggregation_keys': self.aggregation_keys, 'timestamp_key': self.timestamp_key, 'historical_timestamp_key': self.historical_timestamp_key, 'lookback_window_seconds': self.lookback_window_seconds,
                'lookback_window_lag_seconds': self.lookback_window_lag_seconds, 'lookback_count': self.lookback_count, 'lookback_until_position': self.lookback_until_position, 'expression': self.expression, 'group_name': self.group_name}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
