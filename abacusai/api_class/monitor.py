import dataclasses

from .abstract import ApiClass
from .enums import StdDevThresholdType


@dataclasses.dataclass
class ForecastingMonitorConfig(ApiClass):
    id_column: str = dataclasses.field(default=None)
    timestamp_column: str = dataclasses.field(default=None)
    target_column: str = dataclasses.field(default=None)
    start_time: str = dataclasses.field(default=None)
    end_time: str = dataclasses.field(default=None)

    def to_dict(self):
        return {
            'id_column': self.id_column,
            'timestamp_column': self.timestamp_column,
            'target_column': self.target_column,
            'start_time': self.start_time,
            'end_time': self.end_time,
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
