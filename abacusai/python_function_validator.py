import importlib.util
import os
import tempfile
from typing import Any, Dict

import pandas as pd


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
                elif not isinstance(value, pd.DataFrame):
                    raise TypeError(f'Invalid type for feature group argument {argument["name"]}. '
                                    f'It should be either a string or a Pandas DataFrame, '
                                    f'but got {type(value).__name__}')

    # Create a temporary directory to store input values and Python code.
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write the Python function code to a file and import the function.
        module_name = 'validation_module'
        module_path = os.path.join(temp_dir, f'{module_name}.py')

        with open(module_path, 'w') as f:
            f.write(function_metadata.code_source.source_code)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        function = getattr(
            module, function_metadata.function_name or python_function_name)

        # Run the function with the provided arguments.
        try:
            result = function(**kwargs)
        except Exception as e:
            raise Exception(
                f"Error while validating Python function '{python_function_name}': {e}")

        # Return the result and status of the validation.
        return result
