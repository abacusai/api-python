import importlib
import inspect
import json
import os
import re
import string
from textwrap import dedent
from typing import IO, Callable, List


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
        'date': 'object',
        'datetime': 'datetime',
    }

    if isinstance(avro_type, dict):
        if avro_type.get('logicalType') in ['timestamp-micros']:
            avro_type = 'datetime'
        elif avro_type.get('logicalType') in ['date']:
            avro_type = 'date'
        else:
            avro_type = avro_type['type']

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


def load_as_pandas_from_avro_fd(fd: IO):
    import fastavro
    import pandas as pd

    reader = fastavro.reader(fd)
    schema = reader.writer_schema
    col_dtypes = {}
    for field in schema['fields']:
        field_name = field['name']
        field_type = field['type']
        if isinstance(field_type, list):
            field_type = get_non_nullable_type(field_type)
        pandas_dtype = avro_to_pandas_dtype(field_type)
        col_dtypes[field_name] = pandas_dtype
    df_part = pd.DataFrame.from_records(
        [r for r in reader], columns=col_dtypes.keys())

    for col in df_part.columns:
        if col_dtypes[col] == 'datetime':
            df_part[col] = pd.to_datetime(
                df_part[col], errors='coerce')

        if pd.core.dtypes.common.is_datetime64_ns_dtype(df_part[col]):
            df_part[col] = df_part[col].dt.tz_localize(
                None)
        elif str(df_part[col].dtype).lower() != str(col_dtypes[col]).lower():
            df_part[col] = df_part[col].astype(
                col_dtypes[col])
    return df_part


def load_as_pandas_from_avro_files(files: List[str], download_method: Callable, max_workers: int = 10):
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    data_df = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tempfile.TemporaryDirectory() as tmp_dir:
            df_parts = []
            file_futures = [executor.submit(
                download_method, file_part, tmp_dir, i) for i, file_part in enumerate(files)]
            for future in file_futures:
                part_path = future.result()
                with open(part_path, 'rb') as part_data:
                    df_part = load_as_pandas_from_avro_fd(part_data)
                    df_parts.append(df_part)
        data_df = pd.concat(df_parts, ignore_index=True)

    return data_df


class DocstoreUtils:
    """Utility class for loading docstore data.
    Needs to be updated if docstore formats change."""

    DOC_ID = 'doc_id'
    PREDICTION_PREFIX = 'prediction'
    FIRST_PAGE = 'first_page'
    LAST_PAGE = 'last_page'
    PAGE_TEXT = 'page_text'
    PAGES = 'pages'
    TOKENS = 'tokens'
    PAGES_ZIP_METADATA = 'pages_zip_metadata'
    PAGE_DATA = 'page_data'

    @staticmethod
    def get_archive_id(doc_id: str):
        parts = doc_id.split('-')
        if len(parts) < 3:
            raise ValueError(f'Unsupported doc_id: {doc_id}')

        if parts[0] == DocstoreUtils.PREDICTION_PREFIX:
            raise ValueError(f'Unsupported doc_id: {doc_id}')

        dataset_version, archive_number, _ = parts[:3]
        return f'{dataset_version}-{archive_number}'

    @staticmethod
    def get_page_id(doc_id: str, page: int):
        return f'{doc_id}-page-{page}'

    @classmethod
    def get_pandas_pages_df(cls, df, feature_group_version: str, doc_id_column: str, document_column: str,
                            get_docstore_resource_bytes: Callable[..., bytes], max_workers: int = 10):
        from concurrent.futures import ThreadPoolExecutor

        import pandas as pd

        group_by_archive = df.groupby(
            df[doc_id_column].apply(lambda x: cls.get_archive_id(x)))
        load_pages_args = []

        for archive_id, archive_group_df in group_by_archive:
            pages_metadata_bytes = get_docstore_resource_bytes(
                feature_group_version, cls.PAGES_ZIP_METADATA, archive_id)
            pages_metadata = json.loads(pages_metadata_bytes.decode('utf-8'))

            for doc_id, pages_ref in archive_group_df[[doc_id_column, document_column]].values:
                if not pages_ref:
                    continue
                for page in range(pages_ref[cls.FIRST_PAGE], pages_ref[cls.LAST_PAGE] + 1):
                    page_id = cls.get_page_id(doc_id, page)
                    offset, size = pages_metadata[page_id]
                    load_pages_args.append((archive_id, offset, size))

        def load_page(archive_id: str, offset: int, size: int):
            page_bytes = get_docstore_resource_bytes(
                feature_group_version, cls.PAGE_DATA, archive_id, offset=offset, size=size)
            return page_bytes.decode('utf-8')

        with ThreadPoolExecutor(max_workers) as executor:
            pages_list = executor.map(
                lambda args: load_page(*args), load_pages_args)

        pages_df = pd.DataFrame([json.loads(page) for page in pages_list])
        return pages_df

    @classmethod
    def get_pandas_documents_df(cls, df, feature_group_version: str, doc_id_column: str, document_column: str,
                                get_docstore_resource_bytes: Callable, max_workers: int = 10):
        pages_ref_column = f'__{document_column}'
        original_columns = df.columns

        # Re-name page_ids column so that the generated document column does not have the same name
        df = df.rename(columns={document_column: pages_ref_column})
        pages_df = cls.get_pandas_pages_df(df, feature_group_version, doc_id_column, pages_ref_column,
                                           get_docstore_resource_bytes, max_workers)

        # pages_df will have "doc_id" as column name which can be different from doc_id_column
        pages_df = pages_df.rename(columns={cls.DOC_ID: doc_id_column})

        # Convert column with tokens per page (list of list) to column with Document format:
        # {TOKENS: [list of tokens in document], PAGES: [list of pages in document]}.
        # No need to sort as page_df is already sorted.
        def combine_doc_info(group):
            pages = list(group[cls.PAGE_TEXT])
            if cls.TOKENS in group:
                tokens = [tok for page_tokens in group[cls.TOKENS]
                          for tok in page_tokens]
                return {cls.PAGES: pages, cls.TOKENS: tokens}
            return {cls.PAGES: pages}

        doc_infos = pages_df.groupby(doc_id_column).apply(
            combine_doc_info).reset_index(name=document_column)
        document_df = df.merge(doc_infos, on=doc_id_column, how='left')
        document_df = document_df[original_columns]
        return document_df


def try_abacus_internal_copy(src_suffix, dst_local, raise_exception=True):
    """ Retuns true if the file was copied, false otherwise"""
    safe_to_do_internal_copy = os.environ.get(
        'SAFE_TO_DO_INTERNAL_COPY') or 'true'
    if safe_to_do_internal_copy == 'true' and os.environ.get('ABACUS_GENERATED_ARTIFACTS_DIR') and importlib.util.find_spec('abacus_internal') and importlib.util.find_spec('abacus_internal.cloud_copy'):
        try:
            from abacus_internal.cloud_copy import copy_cloud_file_to_local
            copy_cloud_file_to_local(src_suffix, dst_local)
        except Exception:
            if raise_exception:
                raise Exception(
                    'Something went wrong while executing in Abacus environment. Please contact support')
        return True
    return False
