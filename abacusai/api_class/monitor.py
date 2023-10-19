import dataclasses

from .abstract import ApiClass


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
