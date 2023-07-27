import dataclasses
from typing import Any

from . import enums
from .abstract import ApiClass


@dataclasses.dataclass
class PythonFunctionArgument(ApiClass):
    """
    A config class for python function arguments

    Args:
        variable_type (PythonFunctionArgumentType): The type of the python function argument
        name (str): The name of the python function variable
        is_required (bool): Whether the argument is required
        value (Any): The value of the argument
        pipeline_variable (str): The name of the pipeline variable to use as the value
    """
    variable_type: enums.PythonFunctionArgumentType = dataclasses.field(default=None)
    name: str = dataclasses.field(default=None)
    is_required: bool = dataclasses.field(default=True)
    value: Any = dataclasses.field(default=None)
    pipeline_variable: str = dataclasses.field(default=None)


@dataclasses.dataclass
class OutputVariableMapping(ApiClass):
    """
    A config class for python function arguments

    Args:
        variable_type (PythonFunctionOutputArgumentType): The type of the python function output argument
        name (str): The name of the python function variable
    """
    variable_type: enums.PythonFunctionOutputArgumentType = dataclasses.field(default=None)
    name: str = dataclasses.field(default=None)
