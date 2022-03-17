from .return_class import AbstractApiClass


class PointInTimeFeature(AbstractApiClass):
    """
        A point in time feature description

        Args:
            client (ApiClient): An authenticated API Client instance
            historyTableName (str): The table name of the history table. If not specified, we use the current table to do a self join.
            aggregationKeys (list of string): List of keys to use for join the historical table and performing the window aggregation.
            timestampKey (str): Name of feature which contains the timestamp value for the point in time feature
            historicalTimestampKey (str): Name of feature which contains the historical timestamp.
            lookbackWindowSeconds (float): If window is specified in terms of time, number of seconds in the past from the current time for start of the window.
            lookbackWindowLagSeconds (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookbackCount (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookbackUntilPosition (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
            groupName (str): The group name this point in time feature belongs to
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

    def __repr__(self):
        return f"PointInTimeFeature(history_table_name={repr(self.history_table_name)},\n  aggregation_keys={repr(self.aggregation_keys)},\n  timestamp_key={repr(self.timestamp_key)},\n  historical_timestamp_key={repr(self.historical_timestamp_key)},\n  lookback_window_seconds={repr(self.lookback_window_seconds)},\n  lookback_window_lag_seconds={repr(self.lookback_window_lag_seconds)},\n  lookback_count={repr(self.lookback_count)},\n  lookback_until_position={repr(self.lookback_until_position)},\n  expression={repr(self.expression)},\n  group_name={repr(self.group_name)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'history_table_name': self.history_table_name, 'aggregation_keys': self.aggregation_keys, 'timestamp_key': self.timestamp_key, 'historical_timestamp_key': self.historical_timestamp_key, 'lookback_window_seconds': self.lookback_window_seconds, 'lookback_window_lag_seconds': self.lookback_window_lag_seconds, 'lookback_count': self.lookback_count, 'lookback_until_position': self.lookback_until_position, 'expression': self.expression, 'group_name': self.group_name}
