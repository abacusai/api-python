import importlib
import json
import os
import re
import string
import uuid
from enum import Enum
from itertools import groupby
from typing import IO, Callable, List

import pandas as pd


INVALID_PANDAS_COLUMN_NAME_CHARACTERS = '[^A-Za-z0-9_]'


def clean_column_name(column):
    cleaned_col = re.sub(
        INVALID_PANDAS_COLUMN_NAME_CHARACTERS, '_', column).lstrip('_')
    if cleaned_col and cleaned_col[0] not in string.ascii_letters:
        cleaned_col = 'Column_' + cleaned_col
    return cleaned_col


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


def _get_spark_incompatible_columns(df):
    # Spark-compatible pandas dtypes
    spark_compatible_pd_dtypes = {
        'int8', 'int16', 'int32', 'int64',
        'float32', 'float64',
        'bool',           # Standard boolean type
        'boolean',        # Nullable BooleanDtype
        'object',         # Assuming they contain strings
        'string',         # StringDtype introduced in pandas 1.0
        'datetime64[ns]',
        'timedelta64[ns]'
    }

    incompatible_columns = []

    for col in df.columns:
        dtype = df[col].dtype
        dtype_str = str(dtype)
        if pd.api.types.is_extension_array_dtype(dtype):
            dtype_name = dtype.name.lower()
            if dtype_name not in spark_compatible_pd_dtypes:
                incompatible_columns.append((col, dtype_name))
        elif dtype_str not in spark_compatible_pd_dtypes:
            incompatible_columns.append((col, dtype_str))

    return incompatible_columns, spark_compatible_pd_dtypes


def get_non_nullable_type(types):
    non_nullable_types = [
        avro_type for avro_type in types if avro_type != 'null']
    return non_nullable_types[0] if non_nullable_types else None


class StreamingHandler(str):
    def __new__(cls, value, context=None, section_key=None, data_type='text', is_transient=False):
        if context:
            cls.process_streaming_data(
                value, context, section_key, data_type, is_transient)
        return str.__new__(cls, value)

    @classmethod
    def process_streaming_data(cls, value, context, section_key, data_type, is_transient):
        if hasattr(context, 'streamed_section_response') and hasattr(context, 'streamed_response') and not is_transient:
            if data_type == 'text':
                if section_key:
                    entry_exists = False
                    for i, item in enumerate(context.streamed_section_response):
                        if item['id'] == section_key:
                            if ((isinstance(context.streamed_section_response[i]['contents'], str) and isinstance(value, str)) or
                               (isinstance(context.streamed_section_response[i]['contents'], list) and isinstance(value, list))):
                                context.streamed_section_response[i]['contents'] += value
                            else:
                                context.streamed_section_response[i]['contents'] = value
                            entry_exists = True
                            break
                    if not entry_exists:
                        context.streamed_section_response.append(
                            {'id': section_key, 'type': data_type, 'mime_type': 'text/plain', 'contents': value, 'message_id': str(uuid.uuid4())})
                else:
                    context.streamed_response.append(str(value))
            elif data_type == 'segment':
                context.streamed_section_response.append(value)


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


def validate_workflow_node_inputs(nodes_info, agent_workflow_node_id, keyword_arguments: dict, sample_user_inputs: dict, filtered_workflow_vars: dict):
    from .api_class import WorkflowNodeInputType
    input_mappings = nodes_info[agent_workflow_node_id].get(
        'input_mappings', {})
    input_schema = nodes_info[agent_workflow_node_id].get('input_schema', {})
    if input_schema.get('runtime_schema', False):
        keyword_arguments = {input_schema.get(
            'schema_prop'): keyword_arguments}
    for input_mapping in input_mappings:
        input_name = input_mapping['name']
        variable_type = input_mapping['variable_type']
        is_required = input_mapping.get('is_required', True)
        variable_source = input_mapping['variable_source']
        if variable_type == 'WORKFLOW_VARIABLE':
            if variable_source not in filtered_workflow_vars:
                raise ValueError(
                    f'The stage corresponding to "{agent_workflow_node_id}" requires variables from {variable_source} stage which are not there.')
            if input_name not in filtered_workflow_vars[variable_source] and is_required:
                raise ValueError(
                    f'Missing required input "{input_name}" in workflow vars for workflow node "{agent_workflow_node_id}".')
            else:
                keyword_arguments[input_name] = filtered_workflow_vars[variable_source][input_name]
        elif variable_type == WorkflowNodeInputType.USER_INPUT:
            if sample_user_inputs and input_name in sample_user_inputs:
                keyword_arguments[input_name] = sample_user_inputs[input_name]
            elif variable_source in filtered_workflow_vars and input_name in filtered_workflow_vars[variable_source]:
                keyword_arguments[input_name] = filtered_workflow_vars[variable_source][input_name]
            else:
                if is_required:
                    raise ValueError(
                        f'User input for "{input_name}" is required for the "{agent_workflow_node_id}" node.')
                else:
                    keyword_arguments[input_name] = None


def run(nodes: List[dict], primary_start_node: str, graph_info: dict, sample_user_inputs: dict = None, agent_workflow_node_id: str = None, workflow_vars: dict = {}, topological_dfs_stack: List = []):
    from .api_class import WorkflowNodeInputType
    source_code = graph_info['source_code']
    exec(source_code, globals())

    nodes_info: dict = {node['name']: node for node in nodes}
    traversal_orders = graph_info['traversal_orders']
    nodes_ancestors = graph_info['nodes_ancestors']
    nodes_inedges = graph_info['nodes_inedges']
    primary_start_node = primary_start_node or graph_info['default_root_node']

    primary_traversal_order = traversal_orders[primary_start_node]
    run_info = {}
    workflow_vars = workflow_vars.copy()
    next_agent_workflow_node_id = None

    if agent_workflow_node_id:
        next_agent_workflow_node_id = agent_workflow_node_id
        if next_agent_workflow_node_id not in traversal_orders.keys():
            if next_agent_workflow_node_id not in nodes_info:
                raise ValueError(
                    f'The provided workflow node id "{next_agent_workflow_node_id}" is not part of the workflow. Please provide a valid node id.')
            else:
                topological_dfs_stack.append(next_agent_workflow_node_id)
        else:
            topological_dfs_stack = [next_agent_workflow_node_id]
    else:
        next_agent_workflow_node_id = primary_start_node
        topological_dfs_stack = [primary_start_node]

    flow_traversal_order = primary_traversal_order
    for root, traversal_order in traversal_orders.items():
        if next_agent_workflow_node_id in traversal_order:
            flow_traversal_order = traversal_order
            break

    run_history = []
    workflow_node_outputs = {}
    while (True):
        agent_workflow_node_id = next_agent_workflow_node_id
        node_ancestors = nodes_ancestors[agent_workflow_node_id]

        # To ensure the node takes inputs only from it's ancestors.
        # workflow_vars must always contain an entry for ancestor, the error is somewhere else if this ever errors out.
        filtered_workflow_vars = {}
        for ancestor in node_ancestors:
            if ancestor not in workflow_vars:
                raise ValueError(
                    f'Ancestor "{ancestor}" of node "{agent_workflow_node_id}" is not executed yet. Please make sure the ancestor nodes are executed before the current node.')
            else:
                filtered_workflow_vars[ancestor] = workflow_vars[ancestor]

        arguments = []
        keyword_arguments = {}
        validate_workflow_node_inputs(nodes_info, agent_workflow_node_id,
                                      keyword_arguments, sample_user_inputs, filtered_workflow_vars)

        try:
            func = eval(nodes_info[agent_workflow_node_id]['function_name'])
            node_response = func(*arguments, **keyword_arguments)
            workflow_node_outputs[agent_workflow_node_id] = node_response.to_dict(
            )
            node_workflow_vars = process_node_response(node_response)
        except Exception as error:
            raise ValueError(
                f'Error in running workflow node {agent_workflow_node_id}: {error}')

        workflow_vars[agent_workflow_node_id] = node_workflow_vars
        next_agent_workflow_node_id = None
        needs_user_input = False

        potential_next_index = flow_traversal_order.index(
            topological_dfs_stack[-1]) + 1
        potential_next_agent_workflow_node_id = None
        while (potential_next_index < len(flow_traversal_order)):
            potential_next_agent_workflow_node_id = flow_traversal_order[potential_next_index]
            incoming_edges = nodes_inedges[potential_next_agent_workflow_node_id]
            valid_next_node = True
            for source, _, details in incoming_edges:
                if source not in topological_dfs_stack:
                    valid_next_node = False
                    potential_next_index += 1
                    break
                else:
                    edge_evaluate_result = evaluate_edge_condition(
                        source, potential_next_agent_workflow_node_id, details, workflow_vars)
                    if not edge_evaluate_result:
                        valid_next_node = False
                        potential_next_index += 1
                        break
            if valid_next_node:
                next_agent_workflow_node_id = potential_next_agent_workflow_node_id
                break

        if next_agent_workflow_node_id:
            next_node_input_mappings = nodes_info[next_agent_workflow_node_id].get(
                'input_mappings', [])
            needs_user_input = any([input_mapping['variable_type'] ==
                                   WorkflowNodeInputType.USER_INPUT for input_mapping in next_node_input_mappings])

        if needs_user_input:
            run_history.append(
                f'Workflow node {agent_workflow_node_id} completed with next node {next_agent_workflow_node_id} and needs user_inputs')
        else:
            run_history.append(
                f'Workflow node {agent_workflow_node_id} completed with next node {next_agent_workflow_node_id}')
            topological_dfs_stack.append(next_agent_workflow_node_id)

        if next_agent_workflow_node_id is None or needs_user_input:
            break

    run_info['workflow_node_outputs'] = workflow_node_outputs
    run_info['run_history'] = run_history

    workflow_info = {}
    workflow_info['workflow_vars'] = workflow_vars
    workflow_info['topological_dfs_stack'] = topological_dfs_stack
    workflow_info['run_info'] = run_info

    return workflow_info


def evaluate_edge_condition(source, target, details, workflow_vars):
    try:
        condition = details.get('EXECUTION_CONDITION')
        if condition:
            result = execute_python_source(
                condition, workflow_vars.get(source, {}))
            return result
        return True
    except Exception as e:
        raise ValueError(
            f"Error evaluating edge '{source}'-->'{target}': {str(e)}")


def execute_python_source(python_expression, variables):
    try:
        # Evaluate the expression using the variables dictionary
        result = eval(python_expression, {}, variables)
        return result
    except Exception as e:
        # Handle any exceptions that may occur during evaluation
        raise ValueError(f'Error evaluating expression: {e}')


def process_node_response(node_response):
    output_vars = {}
    for variable in node_response.section_data_list:
        for key, value in variable.items():
            output_vars[key] = value
    return output_vars


class StreamType(Enum):
    MESSAGE = 'message'
    SECTION_OUTPUT = 'section_output'
    SEGMENT = 'segment'


class DocstoreUtils:
    """Utility class for loading docstore data.
    Needs to be updated if docstore formats change."""

    DOC_ID = 'doc_id'
    PREDICTION_PREFIX = 'prediction'
    FIRST_PAGE = 'first_page'
    LAST_PAGE = 'last_page'
    PAGE_TEXT = 'page_text'
    PAGES = 'pages'
    CONTENT = 'content'
    TOKENS = 'tokens'
    PAGES_ZIP_METADATA = 'pages_zip_metadata'
    PAGE_DATA = 'page_data'
    HEIGHT = 'height'
    WIDTH = 'width'
    METADATA = 'metadata'
    PAGE = 'page'
    BLOCK = 'block'
    LINE = 'line'
    EXTRACTED_TEXT = 'extracted_text'
    EMBEDDED_TEXT = 'embedded_text'
    PAGE_MARKDOWN = 'page_markdown'
    PAGE_LLM_OCR = 'page_llm_ocr'
    PAGE_TABLE_TEXT = 'page_table_text'
    MARKDOWN_FEATURES = 'markdown_features'
    MULTI_MODE_OCR_TEXT = 'multi_mode_ocr_text'
    DOCUMENT_PROCESSING_CONFIG = 'document_processing_config'
    DOCUMENT_PROCESSING_VERSION = 'document_processing_version'

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

    @staticmethod
    def get_content_hash(doc_id: str):
        parts = doc_id.split('-')
        content_hash = next((part for part in parts if len(part) == 64), None)
        return content_hash

    @classmethod
    def get_pandas_pages_df(cls, df, feature_group_version: str, doc_id_column: str, document_column: str,
                            get_docstore_resource_bytes: Callable[..., bytes], get_document_processing_result_infos: Callable, max_workers: int = 10):
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import pandas as pd

        chunk_size = 10 * 1024 * 1024

        def is_valid_config(x):
            if not isinstance(x, dict):
                return False
            if cls.DOCUMENT_PROCESSING_CONFIG not in x:
                return False
            if x[cls.DOCUMENT_PROCESSING_CONFIG] is None:
                return False
            if x[cls.DOCUMENT_PROCESSING_CONFIG] == {}:
                return False
            # if all keys are None, return False
            if all(v is None for v in x[cls.DOCUMENT_PROCESSING_CONFIG].values()):
                return False
            return True

        pages_df_with_config = None
        df_with_config = df[df[document_column].apply(is_valid_config)]
        df = df[~df[doc_id_column].isin(df_with_config[doc_id_column])]

        if len(df_with_config) > 0:
            doc_ids = df_with_config[doc_id_column].values
            content_hash_to_doc_id = {cls.get_content_hash(
                doc_id): doc_id for doc_id in doc_ids}
            content_hash_list = list(content_hash_to_doc_id.keys())

            unique_document_processing_configs = set(df_with_config[document_column].apply(
                lambda x: str(x.get(cls.DOCUMENT_PROCESSING_CONFIG))))
            if len(unique_document_processing_configs) > 1:
                raise ValueError(
                    'Loading documents with different document processing configs is not supported yet. Please make sure all rows have the same document processing config.')

            sample_page_infos = df_with_config.iloc[0][document_column]
            document_processing_config = sample_page_infos[cls.DOCUMENT_PROCESSING_CONFIG]
            document_processing_version = sample_page_infos.get(
                cls.DOCUMENT_PROCESSING_VERSION)

            page_offsets_and_zip_location_list = get_document_processing_result_infos(
                content_hash_list, document_processing_config, document_processing_version) or []

            zip_location_to_offsets = {}
            for row in page_offsets_and_zip_location_list:
                zip_location_to_offsets.setdefault(
                    row['result_zip_path'], []).append(row)

            pages_list = []
            for result_zip_path, page_offsets in zip_location_to_offsets.items():
                zip_bytes = get_docstore_resource_bytes(
                    feature_group_version, cls.PAGE_DATA, result_zip_path=result_zip_path)
                for row in page_offsets:
                    start_offset, file_size = row['start_offset'], row['file_size']
                    page_content = zip_bytes[start_offset:start_offset + file_size]
                    page_data = json.loads(page_content.decode('utf-8'))
                    pages_list.append((row['content_hash'], page_data))

            json_pages_list = [{**(page or {}), doc_id_column: content_hash_to_doc_id[content_hash]}
                               for content_hash, page in pages_list]
            pages_df_with_config = pd.DataFrame(json_pages_list)
            pages_df_with_config = pages_df_with_config.replace({np.nan: None})

        df = df.drop_duplicates([doc_id_column])
        group_by_archive = df.groupby(
            df[doc_id_column].apply(lambda x: cls.get_archive_id(x)))

        def load_page(archive_bytes: bytes, offset: int, size: int):
            page_bytes = archive_bytes[offset:offset + size]
            return json.loads(page_bytes.decode('utf-8'))

        def download_archive_chunk(offset: int, size: int, archive_id: str):
            return get_docstore_resource_bytes(feature_group_version, cls.PAGE_DATA, archive_id, offset=offset, size=size)

        pages_list = []

        for archive_id, archive_group_df in group_by_archive:
            load_pages_args = []
            min_offset = None
            max_offset = None
            pages_metadata_bytes = get_docstore_resource_bytes(
                feature_group_version, cls.PAGES_ZIP_METADATA, archive_id)
            pages_metadata = json.loads(pages_metadata_bytes.decode('utf-8'))

            for doc_id, pages_ref in archive_group_df[[doc_id_column, document_column]].values:
                if not pages_ref:
                    continue
                for page in range(pages_ref[cls.FIRST_PAGE], pages_ref[cls.LAST_PAGE] + 1):
                    page_id = cls.get_page_id(doc_id, page)
                    offset, size = pages_metadata[page_id]
                    load_pages_args.append((offset, size))
                    if min_offset is None:
                        min_offset = offset
                    if max_offset is None:
                        max_offset = offset + size
                    min_offset = min(min_offset, offset)
                    max_offset = max(max_offset, offset + size)

            chunk_args = [(offset, chunk_size, archive_id)
                          for offset in range(min_offset, max_offset, chunk_size)]

            with ThreadPoolExecutor(max_workers) as executor:
                archive_chunks = executor.map(
                    lambda args: download_archive_chunk(*args), chunk_args)

            archive_bytes = b''.join(archive_chunks)

            for offset, size in load_pages_args:
                pages_list.append(
                    load_page(archive_bytes, offset - min_offset, size))

        pages_df = pd.DataFrame(pages_list)
        pages_df = pages_df.replace({np.nan: None})

        if pages_df_with_config is not None:
            pages_df = pd.concat([pages_df, pages_df_with_config])

        return pages_df

    @classmethod
    def get_pandas_documents_df(cls, df, feature_group_version: str, doc_id_column: str, document_column: str,
                                get_docstore_resource_bytes: Callable, get_document_processing_result_infos: Callable, max_workers: int = 10):
        pages_ref_column = f'__{document_column}'
        original_columns = df.columns

        # Re-name page_ids column so that the generated document column does not have the same name
        df = df.rename(columns={document_column: pages_ref_column})
        pages_df = cls.get_pandas_pages_df(df, feature_group_version, doc_id_column, pages_ref_column,
                                           get_docstore_resource_bytes, get_document_processing_result_infos, max_workers)

        # pages_df will have "doc_id" as column name which can be different from doc_id_column
        pages_df = pages_df.rename(columns={cls.DOC_ID: doc_id_column})

        def tokens_to_pages(tokens: List[dict]) -> List[str]:
            result = []
            if not tokens:
                return result
            for _, page_tokens in groupby(sorted(tokens, key=lambda t: (t[cls.PAGE], t[cls.BLOCK], t[cls.LINE])), key=lambda t: t[cls.PAGE]):
                blocks = []
                for _, block_tokens in groupby(page_tokens, key=lambda t: t[cls.BLOCK]):
                    lines = []
                    for _, line_tokens in groupby(block_tokens, key=lambda t: t[cls.LINE]):
                        lines += [' '.join(t[cls.CONTENT]
                                           for t in line_tokens), '\n']
                    if lines:
                        lines.pop()
                        blocks += [''.join(lines)]
                if blocks:
                    result.append('\n'.join(blocks))
            return result

        def combine_page_texts(pages_text: List[str]) -> str:
            return '\n'.join(pages_text)

        def tokens_to_text(tokens: List[dict]) -> str:
            return combine_page_texts(tokens_to_pages(tokens))

        # Convert column with tokens per page (list of list) to column with Document format:
        # {TOKENS: [list of tokens in document], PAGES: [list of pages in document]}.
        # No need to sort as page_df is already sorted.
        def combine_doc_info(group):
            page_infos = group.to_dict(orient='records')
            document_data = {
                cls.METADATA: [{cls.HEIGHT: page.get(cls.HEIGHT, None), cls.WIDTH: page.get(cls.WIDTH, None), cls.PAGE: page_no, cls.MARKDOWN_FEATURES: page.get(cls.MARKDOWN_FEATURES)}
                               for page_no, page in enumerate(page_infos)],
                cls.TOKENS: [token for page in page_infos for token in page.get(cls.TOKENS) or []],
                # default to embedded text
                cls.PAGES: [page.get(cls.PAGE_TEXT) or '' for page in page_infos],
                **({cls.DOC_ID: page_infos[0][cls.DOC_ID]} if cls.DOC_ID in page_infos[0] else {}),
            }
            document_data[cls.EMBEDDED_TEXT] = combine_page_texts(info.get(
                cls.EMBEDDED_TEXT) or info.get(cls.PAGE_TEXT) or '' for info in page_infos)
            page_texts = None
            for k in [cls.MULTI_MODE_OCR_TEXT, cls.PAGE_MARKDOWN, cls.PAGE_LLM_OCR, cls.PAGE_TABLE_TEXT]:
                if page_infos[0].get(k) and not document_data.get(cls.PAGE_MARKDOWN):
                    document_data[cls.PAGE_MARKDOWN] = page_texts = [
                        page.get(k, '') for page in page_infos]
                    break
            if not page_texts and page_infos[0].get(cls.EXTRACTED_TEXT):
                page_texts = [page.get(cls.EXTRACTED_TEXT, '')
                              for page in page_infos]
            elif not page_texts and page_infos[0].get(cls.TOKENS):
                page_texts = [tokens_to_text(pg[cls.TOKENS])
                              for pg in page_infos]
            if page_texts:
                document_data[cls.PAGES] = page_texts
            page_texts = page_texts or ['' for _ in page_infos]
            document_data[cls.EXTRACTED_TEXT] = combine_page_texts(page_texts)
            return document_data

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
