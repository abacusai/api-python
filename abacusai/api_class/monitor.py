import dataclasses

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
            'window_config': TimeWindowConfig.from_dict(self.window_config).to_dict() if self.window_config else None,
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
            'lower_bound': StdDevThreshold.from_dict(self.lower_bound).to_dict() if self.lower_bound else None,
            'upper_bound': StdDevThreshold.from_dict(self.upper_bound).to_dict() if self.upper_bound else None,
        }
