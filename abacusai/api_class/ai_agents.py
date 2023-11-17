import dataclasses
from typing import Union

from . import enums
from .abstract import ApiClass


@dataclasses.dataclass
class FieldDescriptor(ApiClass):
    """
    Configs for vector store indexing.

    Args:
        field (str): The field to be extracted. This will be used as the key in the response.
        description (str): The description of this field. If not included, the response_field will be used.
        example_extraction (Union[str, int, bool, float]): An example of this extracted field.
        type (enums.FieldDescriptorType): The type of this field. If not provided, the default type is STRING.
    """
    field: str = dataclasses.field()
    description: str = dataclasses.field(default=None)
    example_extraction: Union[str, int, bool, float, list, dict] = dataclasses.field(default=None)
    type: enums.FieldDescriptorType = dataclasses.field(default=enums.FieldDescriptorType.STRING)
