from .code_source import CodeSource
from .return_class import AbstractApiClass


class PythonPlotFunction(AbstractApiClass):
    """
        Create a Plot for a Dashboard

        Args:
            client (ApiClient): An authenticated API Client instance
            notebookId (str): Unique string identifier of the notebook used to spin up the notebook upon creation.
            name (str): The name used to identify the algorithm. Only uppercase letters, numbers, and underscores are allowed.
            createdAt (str): Date and time when the Python function was created, in ISO-8601 format.
            functionVariableMappings (dict): The mappings for function parameters' names.
            functionName (str): The name of the Python function to be used.
            pythonFunctionId (str): Unique string identifier of the Python function.
            functionType (str): The type of the Python function.
            plotName (str): Name of the plot.
            graphReferenceId (str): Reference ID of the dashboard to the plot.
            codeSource (CodeSource): Info about the source code of the Python function.
    """

    def __init__(self, client, notebookId=None, name=None, createdAt=None, functionVariableMappings=None, functionName=None, pythonFunctionId=None, functionType=None, plotName=None, graphReferenceId=None, codeSource={}):
        super().__init__(client, None)
        self.notebook_id = notebookId
        self.name = name
        self.created_at = createdAt
        self.function_variable_mappings = functionVariableMappings
        self.function_name = functionName
        self.python_function_id = pythonFunctionId
        self.function_type = functionType
        self.plot_name = plotName
        self.graph_reference_id = graphReferenceId
        self.code_source = client._build_class(CodeSource, codeSource)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'notebook_id': repr(self.notebook_id), f'name': repr(self.name), f'created_at': repr(self.created_at), f'function_variable_mappings': repr(self.function_variable_mappings), f'function_name': repr(
            self.function_name), f'python_function_id': repr(self.python_function_id), f'function_type': repr(self.function_type), f'plot_name': repr(self.plot_name), f'graph_reference_id': repr(self.graph_reference_id), f'code_source': repr(self.code_source)}
        class_name = "PythonPlotFunction"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'notebook_id': self.notebook_id, 'name': self.name, 'created_at': self.created_at, 'function_variable_mappings': self.function_variable_mappings, 'function_name': self.function_name,
                'python_function_id': self.python_function_id, 'function_type': self.function_type, 'plot_name': self.plot_name, 'graph_reference_id': self.graph_reference_id, 'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
