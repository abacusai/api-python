import inspect
import json
import re
import string
from textwrap import dedent
from typing import Callable


INVALID_PANDAS_COLUMN_NAME_CHARACTERS = '[^A-Za-z0-9_]'


def clean_column_name(column):
    cleaned_col = re.sub(
        INVALID_PANDAS_COLUMN_NAME_CHARACTERS, '_', column).lstrip('_')
    if cleaned_col and cleaned_col[0] not in string.ascii_letters:
        cleaned_col = 'Column_' + cleaned_col
    return cleaned_col


def get_clean_function_source_code(func: Callable):
    source_code = inspect.getsource(func)
    # If function source code has some initial indentation, remove it (Ex - can happen if the functor was defined inside a function)
    source_code = dedent(source_code)
    return source_code


def avro_to_pandas_dtype(avro_type):
    avro_pandas_dtypes = {
        'long': 'Int64',
        'int': 'Int32',
        'float': 'float32',
        'double': 'float64',
        'string': 'object',
        'boolean': 'bool',
        'bytes': 'object',
        'null': 'object',
        'date': 'datetime',
    }

    if isinstance(avro_type, dict):
        avro_type = 'date' if avro_type.get('logicalType') in [
            'date', 'timestamp-micros'] else avro_type['type']

    return avro_pandas_dtypes.get(avro_type, 'object')


def get_non_nullable_type(types):
    non_nullable_types = [
        avro_type for avro_type in types if avro_type != 'null']
    return non_nullable_types[0] if non_nullable_types else None


def get_object_from_context(client, context, variable_name, return_type):
    raw_value = getattr(context, variable_name, None)
    if raw_value is None:
        return None

    from typing import _GenericAlias

    is_container_type = isinstance(return_type, _GenericAlias)
    if not is_container_type and isinstance(raw_value, return_type):
        return raw_value

    typed_value = raw_value

    #
    # Attempt to cast json strings and dicts into api class objects
    #
    try:
        from .return_class import AbstractApiClass

        list_container = return_type is list
        dict_container = return_type is dict
        base_type = return_type
        if is_container_type:
            dict_container = return_type.__origin__ is dict
            list_container = return_type.__origin__ is list
            if hasattr(return_type.__args__[-1], '__bases__'):
                base_type = return_type.__args__[-1]

        is_api_class = issubclass(base_type, AbstractApiClass)
        if isinstance(raw_value, str) and (is_api_class or list_container or dict_container):
            typed_value = json.loads(raw_value)

        if is_api_class:
            if list_container and hasattr(typed_value, '__iter__') and isinstance(next(iter(typed_value)), dict):
                typed_value = [base_type(client, **o) for o in typed_value]
            elif dict_container and isinstance(typed_value, dict) and isinstance(next(iter(typed_value.values())), dict):
                typed_value = {k: base_type(client, **v)
                               for k, v in typed_value.items()}
            elif not list_container and not dict_container and isinstance(typed_value, dict):
                typed_value = base_type(client, **typed_value)

    except Exception:
        pass

    return typed_value
