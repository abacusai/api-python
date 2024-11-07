import ast
import dataclasses
import datetime
import inspect
import re
import sys
from abc import ABC
from copy import deepcopy
from textwrap import dedent
from typing import Any, Callable, Union, _GenericAlias, get_type_hints


if sys.version_info >= (3, 8):
    from typing import get_origin
else:
    from typing_inspect import get_origin

from .enums import ApiEnum


FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


def _validate_instance(value, expected_type):
    if expected_type == callable:
        return callable(value)
    if expected_type is Any:
        return True
    elif isinstance(expected_type, _GenericAlias):
        if expected_type.__origin__ == list:
            if not isinstance(value, list):
                return False
            else:
                for item in value:
                    if not _validate_instance(item, expected_type.__args__[0]):
                        return False
        elif expected_type.__origin__ == Union:
            type_match_found = False
            for possible_type in expected_type.__args__:
                if _validate_instance(value, possible_type):
                    type_match_found = True
                    break
            return type_match_found
        elif expected_type.__origin__ == dict:
            if not isinstance(value, dict):
                return False
            else:
                key_type = expected_type.__args__[0]
                val_type = expected_type.__args__[1]
                for key, val in value.items():
                    if not _validate_instance(key, key_type) or not _validate_instance(val, val_type):
                        return False
    else:
        return isinstance(value, expected_type)
    return True


def _get_user_friendly_type_name(typename):
    type_value = ''
    if isinstance(typename, _GenericAlias):
        if typename.__origin__ == list:
            type_value = f'a list of {_get_user_friendly_type_name(typename.__args__[0])}'
        elif typename.__origin__ == dict:
            type_value += f'a dictionary of key type: {{{_get_user_friendly_type_name(typename.__args__[0])}, {_get_user_friendly_type_name(typename.__args__[1])}}}'
        elif typename.__origin__ == Union:
            type_value += f'one of the following types - ( {", ".join([_get_user_friendly_type_name(possible_type) for possible_type in typename.__args__])} )'
    else:
        type_value = typename.__name__
    return type_value


def validate_class_method_annotations(classname=None, friendly_class_name=None):
    if friendly_class_name is None:
        friendly_class_name = classname

    def validate_types(func):
        nonlocal classname
        if classname is None:
            classname = func.__qualname__.split('.')[0]

        def new_func(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs).arguments
            for keyword, value in list(bound_args.items())[1:]:  # To skip self or cls argument.
                default_value = sig.parameters[keyword].default
                if default_value != sig.parameters[keyword].empty and value == default_value:
                    continue
                expected_type = func.__annotations__.get(keyword)
                if not _validate_instance(value, expected_type):
                    raise ValueError(friendly_class_name, f'Invalid {classname} instance. Argument "{keyword}" must be {_get_user_friendly_type_name(expected_type)}.')
            func(*args, **kwargs)
        return new_func
    return validate_types


def validate_constructor_arg_types(friendly_class_name=None):
    def validate_types(cls):
        cls.__init__ = validate_class_method_annotations(cls.__name__, friendly_class_name)(cls.__init__)
        return cls
    return validate_types


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


def get_clean_function_source_code(func: Callable):
    sample_lambda = (lambda: 0)
    if isinstance(func, type(sample_lambda)) and func.__name__ == sample_lambda.__name__:
        raise ValueError('Lambda function not allowed.')
    source_code = inspect.getsource(func)
    # If function source code has some initial indentation, remove it (Ex - can happen if the functor was defined inside a function)
    source_code = dedent(source_code)
    return source_code


def get_clean_function_source_code_for_agent(func: Callable):
    sample_lambda = (lambda: 0)
    if isinstance(func, type(sample_lambda)) and func.__name__ == sample_lambda.__name__:
        raise ValueError('Lambda function not allowed.')
    source_code = get_source_code(func)
    # If function source code has some initial indentation, remove it (Ex - can happen if the functor was defined inside a function)
    source_code = dedent(source_code)
    return source_code


def get_source_code(func: Callable):
    main_function_name = func.__name__
    source_code = inspect.getsource(func)
    source_code = dedent(source_code)

    function_globals = func.__globals__
    tree = ast.parse(source_code)
    call_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_nodes.append(node.func.id)

    functions_included = {}

    while True:
        if len(call_nodes) == 0:
            break
        new_call_nodes = []
        for function_name in call_nodes:
            if function_name not in functions_included and function_name in function_globals:
                function_callable = function_globals[function_name]
                if inspect.isfunction(function_callable) and function_callable.__module__ == '__main__':
                    cur_node_source_code = inspect.getsource(function_globals[function_name])
                    cur_node_source_code = dedent(cur_node_source_code)
                    functions_included[function_name] = cur_node_source_code
                    cur_node_tree = ast.parse(cur_node_source_code)
                    cur_node_call_nodes = []
                    for node in ast.walk(cur_node_tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                cur_node_call_nodes.append(node.func.id)
                    new_call_nodes.extend(cur_node_call_nodes)
        call_nodes = new_call_nodes

    functions_included.pop(main_function_name, None)
    final_source_code = '\n\n'.join([value for value in functions_included.values()]) + '\n\n' + source_code
    return final_source_code


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
        elif self._support_kwargs and hasattr(self, 'kwargs') and self.kwargs is not None and (self.kwargs.get(item) is not None):
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
        elif self._support_kwargs and hasattr(self, 'kwargs') and self.kwargs is not None and (self.kwargs.get(item) is not None):
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
                kwargs = api_class_dict.get('kwargs', None)
                api_class_dict.update(kwargs or {})
            for k, v in api_class_dict.items():
                if v is not None and k != 'kwargs':
                    if not k.startswith('__'):
                        k = upper_snake_case(k) if self._upper_snake_case_keys else camel_case(k)
                    if isinstance(v, ApiClass):
                        res[k] = to_dict_helper(v)
                    elif isinstance(v, list):
                        res[k] = [to_dict_helper(val) if isinstance(val, ApiClass) else val for val in v]
                    elif isinstance(v, dict):
                        res[k] = {key: to_dict_helper(val) if isinstance(val, ApiClass) else val for key, val in v.items()}
                    elif isinstance(v, datetime.datetime) or isinstance(v, datetime.date):
                        res[k] = v.isoformat() if v else v
                    elif isinstance(v, ApiEnum):
                        res[k] = v.value
                    else:
                        res[k] = v
            return res

        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, input_dict: dict):
        if input_dict is None:
            return None
        obj = None
        if input_dict:
            builder = cls._get_builder()
            if builder:
                config_class_key = None
                value = next((key for key, val in builder.config_class_map.items() if val.__name__ == cls.__name__), None)
                input_dict_with_config_key = input_dict
                if value is not None:
                    input_dict_with_config_key = deepcopy(input_dict)
                    if builder.config_abstract_class._upper_snake_case_keys:
                        config_class_key = upper_snake_case(builder.config_class_key)
                        if config_class_key not in input_dict_with_config_key:
                            input_dict_with_config_key[config_class_key] = value
                    else:
                        config_class_key = builder.config_class_key
                        if config_class_key not in input_dict_with_config_key and camel_case(config_class_key) not in input_dict_with_config_key:
                            input_dict_with_config_key[config_class_key] = value

                obj = builder.from_dict(input_dict_with_config_key)

            if not cls._upper_snake_case_keys:
                input_dict = {snake_case(k): v for k, v in input_dict.items()}
            if not cls._support_kwargs:
                # only use keys that are valid fields in the ApiClass
                field_names = set((field.name) for field in dataclasses.fields(type(obj) if obj else cls))
                input_dict = {k: v for k, v in input_dict.items() if k in field_names}
        if obj is None:
            obj = cls(**input_dict)

        for attr_name, attr_type in get_type_hints(type(obj)).items():
            if attr_name in input_dict and isinstance(input_dict[attr_name], dict) and inspect.isclass(attr_type) and issubclass(attr_type, ApiClass):
                setattr(obj, attr_name, attr_type.from_dict(input_dict[attr_name]))
            elif attr_name in input_dict and get_origin(attr_type) is list and attr_type.__args__ and inspect.isclass(attr_type.__args__[0]) and issubclass(attr_type.__args__[0], ApiClass):
                class_type = attr_type.__args__[0]
                if isinstance(input_dict[attr_name], list):
                    if not all(isinstance(item, dict) or isinstance(item, class_type) for item in input_dict[attr_name]):
                        raise ValueError(attr_name, f'Expected list of {class_type} or dictionary for {attr_name}')
                    setattr(obj, attr_name, [item if isinstance(item, class_type) else class_type.from_dict(item) for item in input_dict[attr_name]])
                else:
                    raise ValueError(attr_name, f'Expected list for {attr_name} but got {type(input_dict[attr_name])}')

        return obj


@dataclasses.dataclass
class _ApiClassFactory(ABC):
    config_abstract_class = None
    config_class_key = None
    config_class_map = {}

    @classmethod
    def from_dict(cls, config: dict) -> ApiClass:
        support_kwargs = cls.config_abstract_class and cls.config_abstract_class._support_kwargs
        is_upper_snake_case_keys = cls.config_abstract_class and cls.config_abstract_class._upper_snake_case_keys
        config_class_key = upper_snake_case(cls.config_class_key) if is_upper_snake_case_keys else cls.config_class_key
        # Logic here is that the we keep the config_class_key in snake_case if _upper_snake_case_keys False else we convert it to upper_snake_case
        # if _upper_snake_case_keys is False then we check in both casing: 1. snake_case and 2. camel_case
        if not is_upper_snake_case_keys and config_class_key not in config and camel_case(config_class_key) in config:
            config_class_key = camel_case(config_class_key)

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
                        trimmed_config[snake_case(k)] = v
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
        config.pop(snake_case(config_class_key), None)

        supported_fields = set([field.name for field in dataclasses.fields(config_class)])
        actual_fields = set(snake_case(key) for key in config.keys())
        if not actual_fields.issubset(supported_fields):
            raise ValueError(f'Unknown fields for {config_class_type}: {actual_fields - supported_fields}')

        return config_class(**config)
