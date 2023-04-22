import inspect
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
        avro_type = 'date' if avro_type.get(
            'logicalType') == 'date' else avro_type['type']

    return avro_pandas_dtypes.get(avro_type, 'object')


def get_non_nullable_type(types):
    non_nullable_types = [
        avro_type for avro_type in types if avro_type != 'null']
    return non_nullable_types[0] if non_nullable_types else None
