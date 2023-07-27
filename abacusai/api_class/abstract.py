import dataclasses
import datetime
import inspect
import re
from abc import ABC

from .enums import ApiEnum


FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def camel_case(value):
    if value:
        components = value.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])
    return value


def upper_snake_case(value):
    if value:
        s1 = FIRST_CAP_RE.sub(r'\1_\2', value)
        return ALL_CAP_RE.sub(r'\1_\2', s1).upper()
    return value


def snake_case(value):
    if value:
        s1 = FIRST_CAP_RE.sub(r'\1_\2', value)
        return ALL_CAP_RE.sub(r'\1_\2', s1).lower()
    return value


@dataclasses.dataclass
class ApiClass(ABC):
    _upper_snake_case_keys: bool = dataclasses.field(default=False, repr=False, init=False)
    _support_kwargs: bool = dataclasses.field(default=False, repr=False, init=False)

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
            api_class_dict = vars(api_class_obj)
            if self._support_kwargs:
                kwargs = api_class_dict.pop('kwargs', None)
                api_class_dict.update(kwargs or {})
            for k, v in api_class_dict.items():
                if not k.startswith('__'):
                    k = upper_snake_case(k) if self._upper_snake_case_keys else camel_case(k)
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
                            res[k] = v.value
                        else:
                            res[k] = v
            return res

        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, input_dict: dict):
        if not cls._upper_snake_case_keys:
            input_dict = {snake_case(k): v for k, v in input_dict.items()}
        return cls(**input_dict)


@dataclasses.dataclass
class _ApiClassFactory(ABC):
    config_abstract_class = None
    config_class_key = None
    config_class_map = {}

    @classmethod
    def from_dict(cls, config: dict) -> ApiClass:
        support_kwargs = cls.config_abstract_class and cls.config_abstract_class._support_kwargs
        config_class_key = cls.config_class_key if (cls.config_abstract_class and not cls.config_abstract_class._upper_snake_case_keys) else camel_case(cls.config_class_key)
        if not support_kwargs and config_class_key not in (config or {}):
            raise KeyError(f'Could not find {config_class_key} in {config}')
        config_class_type = config.get(config_class_key, None)
        if isinstance(config_class_type, str):
            config_class_type = config_class_type.upper()
        config_class = cls.config_class_map.get(config_class_type)
        if support_kwargs:
            if config_class:
                field_names = set((field.name) for field in dataclasses.fields(config_class))
                trimmed_config = {}
                kwargs = {}
                for k, v in config.items():
                    if snake_case(k) in field_names:
                        trimmed_config[k] = v
                    else:
                        kwargs[k] = v
                if len(kwargs):
                    trimmed_config['kwargs'] = kwargs
                config = trimmed_config
            else:
                config = {'kwargs': config}
                config_class = cls.config_abstract_class
        if config_class is None:
            raise ValueError(f'Invalid type {config_class_type}')

        return config_class.from_dict(config)
