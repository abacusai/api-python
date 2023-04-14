from dataclasses import dataclass, field

from .abstract import ApiClass


@dataclass
class ParsingConfig(ApiClass):
    escape: str = field(default='"')
    csv_delimiter: str = field(default=None)
    file_path_with_schema: str = field(default=None)
