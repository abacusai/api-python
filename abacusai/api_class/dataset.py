from dataclasses import dataclass, field

from .abstract import ApiClass


@dataclass
class ParsingConfig(ApiClass):
    escape: str = field(default='"')
