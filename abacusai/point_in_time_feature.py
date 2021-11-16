from .return_class import AbstractApiClass


class PointInTimeFeature(AbstractApiClass):
    """
        A point in time feature description
    """

    def __init__(self, client, historyTableName=None, aggregationKeys=None, timestampKey=None, historicalTimestampKey=None, lookbackWindowSeconds=None, lookbackWindowLagSeconds=None, lookbackCount=None, lookbackUntilPosition=None, expression=None):
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

    def __repr__(self):
        return f"PointInTimeFeature(history_table_name={repr(self.history_table_name)},\n  aggregation_keys={repr(self.aggregation_keys)},\n  timestamp_key={repr(self.timestamp_key)},\n  historical_timestamp_key={repr(self.historical_timestamp_key)},\n  lookback_window_seconds={repr(self.lookback_window_seconds)},\n  lookback_window_lag_seconds={repr(self.lookback_window_lag_seconds)},\n  lookback_count={repr(self.lookback_count)},\n  lookback_until_position={repr(self.lookback_until_position)},\n  expression={repr(self.expression)})"

    def to_dict(self):
        return {'history_table_name': self.history_table_name, 'aggregation_keys': self.aggregation_keys, 'timestamp_key': self.timestamp_key, 'historical_timestamp_key': self.historical_timestamp_key, 'lookback_window_seconds': self.lookback_window_seconds, 'lookback_window_lag_seconds': self.lookback_window_lag_seconds, 'lookback_count': self.lookback_count, 'lookback_until_position': self.lookback_until_position, 'expression': self.expression}
