import dataclasses
from typing import List

from .abstract import ApiClass
from .enums import StdDevThresholdType


@dataclasses.dataclass
class TimeWindowConfig(ApiClass):
    """
    Time Window Configuration

    Args:
        window_duration (int): The duration of the window.
        window_from_start (bool): Whether the window should be from the start of the time series.
    """
    window_duration: int = dataclasses.field(default=None)
    window_from_start: bool = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'window_duration': self.window_duration,
            'window_from_start': self.window_from_start,
        }


@dataclasses.dataclass
class ForecastingMonitorConfig(ApiClass):
    """
    Forecasting Monitor Configuration

    Args:
        id_column (str): The name of the column that contains the unique identifier for the time series.
        timestamp_column (str): The name of the column that contains the timestamp for the time series.
        target_column (str): The name of the column that contains the target value for the time series.
        start_time (str): The start time of the time series data.
        end_time (str): The end time of the time series data.
        window_config (TimeWindowConfig): The windowing configuration for the time series data.
    """
    id_column: str = dataclasses.field(default=None)
    timestamp_column: str = dataclasses.field(default=None)
    target_column: str = dataclasses.field(default=None)
    start_time: str = dataclasses.field(default=None)
    end_time: str = dataclasses.field(default=None)
    window_config: TimeWindowConfig = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'id_column': self.id_column,
            'timestamp_column': self.timestamp_column,
            'target_column': self.target_column,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'window_config': self.window_config.to_dict() if self.window_config else None,
        }


@dataclasses.dataclass
class StdDevThreshold(ApiClass):
    """
    Std Dev Threshold types

    Args:
        threshold_type (StdDevThresholdType): Type of threshold to apply to the item attributes.
        value (float): Value to use for the threshold.
    """
    threshold_type: StdDevThresholdType = dataclasses.field(default=None)
    value: float = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'threshold_type': self.threshold_type,
            'value': self.value,
        }


@dataclasses.dataclass
class ItemAttributesStdDevThreshold(ApiClass):
    """
    Item Attributes Std Dev Threshold for Monitor Alerts

    Args:
        lower_bound (StdDevThreshold): Lower bound for the item attributes.
        upper_bound (StdDevThreshold): Upper bound for the item attributes.
    """
    lower_bound: StdDevThreshold = dataclasses.field(default=None)
    upper_bound: StdDevThreshold = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'lower_bound': self.lower_bound.to_dict() if self.lower_bound else None,
            'upper_bound': self.upper_bound.to_dict() if self.upper_bound else None,
        }


@dataclasses.dataclass
class RestrictFeatureMappings(ApiClass):
    """
    Restrict Feature Mappings for Monitor Filtering

    Args:
        feature_name (str): The name of the feature to restrict the monitor to.
        restricted_feature_values (list): The values of the feature to restrict the monitor to if feature is a categorical.
        start_time (str): The start time of the timestamp feature to filter from
        end_time (str): The end time of the timestamp feature to filter until
        min_value (float): Value to filter the numerical feature above
        max_value (float): Filtering the numerical feature to below this value
    """
    feature_name: str = dataclasses.field(default=None)
    restricted_feature_values: list = dataclasses.field(default_factory=list)
    start_time: str = dataclasses.field(default=None)
    end_time: str = dataclasses.field(default=None)
    min_value: float = dataclasses.field(default=None)
    max_value: float = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'feature_name': self.feature_name,
            'restricted_feature_values': self.restricted_feature_values,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'min_value': self.min_value,
            'max_value': self.max_value,
        }


@dataclasses.dataclass
class MonitorFilteringConfig(ApiClass):
    """
    Monitor Filtering Configuration

    Args:
        start_time (str): The start time of the prediction time col
        end_time (str): The end time of the prediction time col
        restrict_feature_mappings (RestrictFeatureMappings): The feature mapping to restrict the monitor to.
        target_class (str): The target class to restrict the monitor to.
        train_target_feature (str): Set the target feature for the training data.
        prediction_target_feature (str): Set the target feature for the prediction data.
    """
    start_time: str = dataclasses.field(default=None)
    end_time: str = dataclasses.field(default=None)
    restrict_feature_mappings: List[RestrictFeatureMappings] = dataclasses.field(default=None)
    target_class: str = dataclasses.field(default=None)
    train_target_feature: str = dataclasses.field(default=None)
    prediction_target_feature: str = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'restrict_feature_mappings': [item.to_dict() for item in self.restrict_feature_mappings] if self.restrict_feature_mappings else None,
            'target_class': self.target_class,
            'train_target_feature': self.train_target_feature if self.train_target_feature else None,
            'prediction_target_feature': self.prediction_target_feature if self.prediction_target_feature else None,
        }
