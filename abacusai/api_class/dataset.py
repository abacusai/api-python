from dataclasses import dataclass, field

from api_class.abstract import ApiClass


@dataclass
class ParsingConfig(ApiClass):
    escape: str = field(default='"')
