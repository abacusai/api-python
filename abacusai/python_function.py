from typing import List

from .code_source import CodeSource
from .return_class import AbstractApiClass


class PythonFunction(AbstractApiClass):
    """
        Customer created python function

        Args:
            client (ApiClient): An authenticated API Client instance
            notebookId (str): The unique identifier of the notebook used to spin up the notebook upon creation.
            name (str): The name to identify the algorithm, only uppercase letters, numbers, and underscores allowed (i.e. it must be a valid Python identifier)
            createdAt (str): The ISO-8601 string representing when the Python function was created.
            functionVariableMappings (dict): A description of the function variables.
            outputVariableMappings (dict): A description of the variables returned by the function
            functionName (str): The name of the Python function to be used.
            pythonFunctionId (str): The unique identifier of the Python function.
            functionType (str): The type of the Python function.
            packageRequirements (list): The pip package dependencies required to run the code
            description (str): Description of the Python function.
            examples (dict[str, list[str]]): Dictionary containing example use cases and anti-patterns. Includes 'positive' examples showing recommended usage and 'negative' examples showing cases to avoid.
            connectors (dict): Dictionary containing user-level and organization-level connectors
            configurations (dict): Dictionary containing configurations for the Python function
            codeSource (CodeSource): Information about the source code of the Python function.
    """

    def __init__(self, client, notebookId=None, name=None, createdAt=None, functionVariableMappings=None, outputVariableMappings=None, functionName=None, pythonFunctionId=None, functionType=None, packageRequirements=None, description=None, examples=None, connectors=None, configurations=None, codeSource={}):
        super().__init__(client, pythonFunctionId)
        self.notebook_id = notebookId
        self.name = name
        self.created_at = createdAt
        self.function_variable_mappings = functionVariableMappings
        self.output_variable_mappings = outputVariableMappings
        self.function_name = functionName
        self.python_function_id = pythonFunctionId
        self.function_type = functionType
        self.package_requirements = packageRequirements
        self.description = description
        self.examples = examples
        self.connectors = connectors
        self.configurations = configurations
        self.code_source = client._build_class(CodeSource, codeSource)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'notebook_id': repr(self.notebook_id), f'name': repr(self.name), f'created_at': repr(self.created_at), f'function_variable_mappings': repr(self.function_variable_mappings), f'output_variable_mappings': repr(self.output_variable_mappings), f'function_name': repr(self.function_name), f'python_function_id': repr(
            self.python_function_id), f'function_type': repr(self.function_type), f'package_requirements': repr(self.package_requirements), f'description': repr(self.description), f'examples': repr(self.examples), f'connectors': repr(self.connectors), f'configurations': repr(self.configurations), f'code_source': repr(self.code_source)}
        class_name = "PythonFunction"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'notebook_id': self.notebook_id, 'name': self.name, 'created_at': self.created_at, 'function_variable_mappings': self.function_variable_mappings, 'output_variable_mappings': self.output_variable_mappings, 'function_name': self.function_name, 'python_function_id': self.python_function_id,
                'function_type': self.function_type, 'package_requirements': self.package_requirements, 'description': self.description, 'examples': self.examples, 'connectors': self.connectors, 'configurations': self.configurations, 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def add_graph_to_dashboard(self, graph_dashboard_id: str, function_variable_mappings: List = None, name: str = None):
        """
        Add a python plot function to a dashboard

        Args:
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to update.
            function_variable_mappings (List): List of arguments to be supplied to the function as parameters, in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].
            name (str): Name of the added python plot

        Returns:
            GraphDashboard: An object describing the graph dashboard.
        """
        return self.client.add_graph_to_dashboard(self.python_function_id, graph_dashboard_id, function_variable_mappings, name)

    def validate_locally(self, kwargs: dict = None) -> any:
        """
        Validates a Python function by running it with the given input values in an local environment. Taking Input Feature Group as either name(string) or Pandas DataFrame in kwargs.

        Args:
            kwargs (dict): A dictionary mapping function arguments to values to pass to the function. Feature group names will automatically be converted into pandas dataframes.

        Returns:
            any: The result of executing the python function

        Raises:
            TypeError: If an Input Feature Group argument has an invalid type or argument is missing.
            Exception: If an error occurs while validating the Python function.
        """
        from .python_function_validator import validate_function_locally
        return validate_function_locally(self.client, self.name, kwargs)
