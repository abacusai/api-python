from .point_in_time_group_feature import PointInTimeGroupFeature
from .return_class import AbstractApiClass


class PointInTimeGroup(AbstractApiClass):
    """
        A point in time group containing point in time features

        Args:
            client (ApiClient): An authenticated API Client instance
            groupName (str): The name of the point in time group
            windowKey (str): Name of feature which contains the timestamp value for the point in time feature
            aggregationKeys (list): List of keys to use for join the historical table and performing the window aggregation.
            lookbackWindow (float): Number of seconds in the past from the current time for start of the window.
            lookbackWindowLag (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookbackCount (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookbackUntilPosition (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.
            historyTableName (str): The table to use for aggregating, if not provided, the source table will be used
            historyWindowKey (str): Name of feature to use for ordering the rows on the history table. If not provided, the windowKey from the source table will be used
            historyAggregationKeys (list): List of keys to use for join the historical table and performing the window aggregation. If not provided, the aggregationKeys from the source table will be used. Must be the same length and order as the source table's aggregationKeys
            features (PointInTimeGroupFeature): 
    """

    def __init__(self, client, groupName=None, windowKey=None, aggregationKeys=None, lookbackWindow=None, lookbackWindowLag=None, lookbackCount=None, lookbackUntilPosition=None, historyTableName=None, historyWindowKey=None, historyAggregationKeys=None, features={}):
        super().__init__(client, None)
        self.group_name = groupName
        self.window_key = windowKey
        self.aggregation_keys = aggregationKeys
        self.lookback_window = lookbackWindow
        self.lookback_window_lag = lookbackWindowLag
        self.lookback_count = lookbackCount
        self.lookback_until_position = lookbackUntilPosition
        self.history_table_name = historyTableName
        self.history_window_key = historyWindowKey
        self.history_aggregation_keys = historyAggregationKeys
        self.features = client._build_class(PointInTimeGroupFeature, features)

    def __repr__(self):
        return f"PointInTimeGroup(group_name={repr(self.group_name)},\n  window_key={repr(self.window_key)},\n  aggregation_keys={repr(self.aggregation_keys)},\n  lookback_window={repr(self.lookback_window)},\n  lookback_window_lag={repr(self.lookback_window_lag)},\n  lookback_count={repr(self.lookback_count)},\n  lookback_until_position={repr(self.lookback_until_position)},\n  history_table_name={repr(self.history_table_name)},\n  history_window_key={repr(self.history_window_key)},\n  history_aggregation_keys={repr(self.history_aggregation_keys)},\n  features={repr(self.features)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'group_name': self.group_name, 'window_key': self.window_key, 'aggregation_keys': self.aggregation_keys, 'lookback_window': self.lookback_window, 'lookback_window_lag': self.lookback_window_lag, 'lookback_count': self.lookback_count, 'lookback_until_position': self.lookback_until_position, 'history_table_name': self.history_table_name, 'history_window_key': self.history_window_key, 'history_aggregation_keys': self.history_aggregation_keys, 'features': self._get_attribute_as_dict(self.features)}
