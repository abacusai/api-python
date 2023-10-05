import dataclasses

from .abstract import ApiClass


@dataclasses.dataclass
class ForecastingMonitorConfig(ApiClass):
    id_column: str = dataclasses.field(default=None)
    timestamp_column: str = dataclasses.field(default=None)
    target_column: str = dataclasses.field(default=None)
    start_time: str = dataclasses.field(default=None)
    end_time: str = dataclasses.field(default=None)
