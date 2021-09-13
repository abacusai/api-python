

class PointInTimeFeature():
    '''
        A point in time feature description
    '''

    def __init__(self, client, historyTableName=None, aggregationKeys=None, timestampKey=None, historicalTimestampKey=None, lookbackWindowSeconds=None, lookbackWindowLagSeconds=None, lookbackCount=None, lookbackUntilPosition=None, expression=None):
        self.client = client
        self.id = None
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
        return f"PointInTimeFeature(history_table_name={repr(self.history_table_name)}, aggregation_keys={repr(self.aggregation_keys)}, timestamp_key={repr(self.timestamp_key)}, historical_timestamp_key={repr(self.historical_timestamp_key)}, lookback_window_seconds={repr(self.lookback_window_seconds)}, lookback_window_lag_seconds={repr(self.lookback_window_lag_seconds)}, lookback_count={repr(self.lookback_count)}, lookback_until_position={repr(self.lookback_until_position)}, expression={repr(self.expression)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'history_table_name': self.history_table_name, 'aggregation_keys': self.aggregation_keys, 'timestamp_key': self.timestamp_key, 'historical_timestamp_key': self.historical_timestamp_key, 'lookback_window_seconds': self.lookback_window_seconds, 'lookback_window_lag_seconds': self.lookback_window_lag_seconds, 'lookback_count': self.lookback_count, 'lookback_until_position': self.lookback_until_position, 'expression': self.expression}
