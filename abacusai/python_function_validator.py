import importlib.util
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from pandas import DataFrame


def validate_function_locally(client, python_function_name: str, kwargs: Dict = None) -> Any:
    """
    Validates a Python function by running it with the given input values in an local environment. Taking Input Feature Group as either name(string) or Pandas DataFrame in kwargs.

    Args:
        client (ApiClient): The AbacusAI client.
        python_function_name (str): The name of the Python function registered in Abacus.AI to execute.
        kwargs (dict): A dictionary mapping function arguments to values to pass to the function. Feature group names will automatically be converted into pandas dataframes.

    Returns:
        any: The result of executing the python function

    Raises:
        TypeError: If an Input Feature Group argument has an invalid type or argument is missing.
        Exception: If an error occurs while validating the Python function.
    """
    kwargs = kwargs or {}
    # Get the function metadata from the AbacusAI client.
    function_metadata = client.describe_python_function(python_function_name)
    for argument in function_metadata.function_variable_mappings:
        # Check that all required arguments are present.
        if argument['name'] not in kwargs and argument['is_required']:
            raise TypeError(f"Missing required argument: '{argument['name']}'")

        # Convert feature group arguments from table name to Pandas DataFrame.
        if argument['variable_type'] == 'FEATURE_GROUP':
            value = kwargs.get(argument['name'])
            if value is not None:
                if isinstance(value, str):
                    feature_group = client.describe_feature_group_by_table_name(
                        value)
                    kwargs[argument['name']] = feature_group.load_as_pandas()
                elif not isinstance(value, DataFrame):
                    raise TypeError(f'Invalid type for feature group argument {argument["name"]}. '
                                    f'It should be either a string or a Pandas DataFrame, '
                                    f'but got {type(value).__name__}')

    # Create a temporary directory to store input values and Python code.
    with tempfile.TemporaryDirectory() as temp_dir:
        py_fun_source_code = function_metadata.code_source.source_code

        # Retrieving Dependent Modules
        dependent_modules = function_metadata.code_source.module_dependencies
        modules_and_source_code = []
        if dependent_modules:
            for module in dependent_modules:
                try:
                    modules_and_source_code.append(
                        client.describe_module(module))
                except Exception as e:
                    logging.exception(
                        f'Error while retrieving dependent module "{module}"')
                    raise e

        for dependent_module in modules_and_source_code:
            dependent_module_path = os.path.join(
                temp_dir, f'{dependent_module.name}.py')
            with open(dependent_module_path, 'w') as f:
                f.write(dependent_module.code_source.source_code)

        # Write the Python function code to a file and import the function.
        module_name = 'validation_module'
        module_path = os.path.join(temp_dir, f'{module_name}.py')

        with open(module_path, 'w') as f:
            f.write(py_fun_source_code)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(
            module, function_metadata.function_name or python_function_name)

        with _MonkeyPatch().context() as m:
            m.syspath_prepend(temp_dir)
            try:
                result = function(**kwargs)
                return result
            except Exception as e:
                logging.exception(
                    f'Error while validating Python function "{python_function_name}"')
                raise e


class _MonkeyPatch:
    """
    Helper class to prepend to ``sys.path`` and undo monkeypatching of attributes
        :syspath_prepend: prepend to ``sys.path`` list of import locations
        :undo: undo all changes made
    """

    def __init__(self) -> None:
        self._savesyspath: Optional[List[str]] = None

    @classmethod
    @contextmanager
    def context(cls) -> Generator['_MonkeyPatch', None, None]:
        m = cls()
        try:
            yield m
        finally:
            m.undo()

    def syspath_prepend(self, path) -> None:
        """
        Prepend ``path`` to ``sys.path`` list of import locations.
        """

        if self._savesyspath is None:
            self._savesyspath = sys.path[:]
        sys.path.insert(0, str(path))

        # this is only needed when pkg_resources was already loaded by the namespace package
        if 'pkg_resources' in sys.modules:
            from pkg_resources import fixup_namespace_packages
            fixup_namespace_packages(str(path))
        from importlib import invalidate_caches

        invalidate_caches()

    def undo(self) -> None:
        """
        Undo all monkeypatching done by this object.
        """
        if self._savesyspath is not None:
            sys.path[:] = self._savesyspath
            self._savesyspath = None
