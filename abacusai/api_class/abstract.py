import dataclasses
import datetime
import inspect
import re
from abc import ABC
from typing import Any

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

    @classmethod
    def _get_builder(cls):
        return None

    def __str__(self):
        non_none_fields = {key: value for key, value in self.__dict__.items() if value is not None}
        field_str = ', '.join([f'{key}={value!r}' for key, value in non_none_fields.items()])
        return f'{self.__class__.__name__}({field_str})'

    def _repr_html_(self):
        return self.__str__()

    def __getitem__(self, item: str):
        if hasattr(self, item):
            return getattr(self, item)
        elif hasattr(self, snake_case(item)):
            return getattr(self, snake_case(item))
        elif self._support_kwargs and (self.kwargs.get(item) is not None):
            return self.kwargs.get(item)
        else:
            raise KeyError(f'Key: {item} is not a property of {self.__class__.__name__}')

    def __setitem__(self, item: str, value: Any):
        if hasattr(self, item):
            setattr(self, item, value)
        elif hasattr(self, snake_case(item)):
            setattr(self, snake_case(item), value)
        elif self._support_kwargs:
            self.kwargs[item] = value
        else:
            raise KeyError(f'Key: {item} is not a property of {self.__class__.__name__}')

    def _unset_item(self, item: str):
        if hasattr(self, item):
            setattr(self, item, None)
        elif hasattr(self, snake_case(item)):
            setattr(self, snake_case(item), None)
        elif self._support_kwargs and (self.kwargs.get(item) is not None):
            self.kwargs.pop(item)
        else:
            raise KeyError(f'Key: {item} is not a property of {self.__class__.__name__}')

    def get(self, item: str, default: Any = None):
        try:
            return self.__getitem__(item)
        except KeyError:
            return default

    def pop(self, item: str, default: Any = NotImplemented):
        try:
            value = self.__getitem__(item)
            self._unset_item(item)
            return value
        except KeyError:
            if default is NotImplemented:
                raise
            return default

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
        if input_dict:
            if builder := cls._get_builder():
                return builder.from_dict(input_dict)
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
            if camel_case(config_class_key) in (config or {}):
                config_class_key = camel_case(config_class_key)
            else:
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
        if not config_class._upper_snake_case_keys:
            config = {snake_case(k): v for k, v in config.items()}
        return config_class(**config)
