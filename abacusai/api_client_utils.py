import re


INVALID_PANDAS_COLUMN_NAME_CHARACTERS = '[^A-Za-z0-9_]'


def clean_column_name(column):
    return re.sub(INVALID_PANDAS_COLUMN_NAME_CHARACTERS, '_', column)
