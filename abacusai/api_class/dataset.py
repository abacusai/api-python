import dataclasses

from .abstract import ApiClass


@dataclasses.dataclass
class ParsingConfig(ApiClass):
    escape: str = dataclasses.field(default='"')
    csv_delimiter: str = dataclasses.field(default=None)
    file_path_with_schema: str = dataclasses.field(default=None)
