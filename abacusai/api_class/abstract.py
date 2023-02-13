import dataclasses
import datetime
import inspect
from abc import ABC

from .enums import ApiEnum


def camel_case(value):
    if value:
        components = value.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    return value


@dataclasses.dataclass
class ApiClass(ABC):

    def __post_init__(self):
        if inspect.isabstract(self):
            raise ValueError(f'Cannot instantiate abstract class {self.__class__.__name__}')

    def to_dict(self):
        """
        Standardizes converting an ApiClass to dictionary.
        Keys of response dictionary are converted to camel case.
        This also validates the fields ( type, value, etc ) received in the dictionary.
        """
        def to_dict_helper(api_class_obj):
            res = {}
            for k, v in vars(api_class_obj).items():
                k = camel_case(k)
                if v is not None:
                    if isinstance(v, ApiClass):
                        res[k] = to_dict_helper(v)
                    elif isinstance(v, list):
                        res[k] = [to_dict_helper(val) if isinstance(val, ApiClass) else val for val in v]
                    elif isinstance(v, dict):
                        res[k] = {key: to_dict_helper(val) if isinstance(val, ApiClass) else val for key, val in v.items()}
                    elif isinstance(v, datetime.datetime) or isinstance(v, datetime.date):
                        res[k] = v.isoformat() if v else v
                    else:
                        if isinstance(v, ApiEnum):
                            res[k] = v.value.upper()
                        else:
                            res[k] = v
            return res

        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, input_dict: dict):
        return cls(**input_dict)


@dataclasses.dataclass
class _ApiClassFactory(ABC):
    config_abstract_class = None
    config_class_key = None
    config_class_map = {}

    @classmethod
    def from_dict(cls, config: dict) -> ApiClass:
        if not config or cls.config_class_key not in config:
            raise KeyError(f'Could not find {camel_case(cls.config_class_key)} in {config or ""}')
        config_class = config.pop(cls.config_class_key, None)
        if isinstance(config_class, str):
            config_class = config_class.upper()
        if not cls.config_class_map.get(config_class):
            raise ValueError(f'Invalid type {config_class}')
        return cls.config_class_map[config_class].from_dict(config)
