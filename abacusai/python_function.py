from .code_source import CodeSource
from .return_class import AbstractApiClass


class PythonFunction(AbstractApiClass):
    """
        Customer created python function

        Args:
            client (ApiClient): An authenticated API Client instance
            notebookId (str): The unique identifier of the notebook used to spin up the notebook upon creation
            name (str): The name to identify the algorithm, only uppercase letters, numbers and underscore allowed
            createdAt (str): When the python function was created
            functionVariableMappings (dict): 
            functionName (str): The name of the python function to be used
            pythonFunctionId (str): The unique identifier of the python function
            functionType (str): The type of the python function
            codeSource (CodeSource): Info about the source code of the python function
    """

    def __init__(self, client, notebookId=None, name=None, createdAt=None, functionVariableMappings=None, functionName=None, pythonFunctionId=None, functionType=None, codeSource={}):
        super().__init__(client, pythonFunctionId)
        self.notebook_id = notebookId
        self.name = name
        self.created_at = createdAt
        self.function_variable_mappings = functionVariableMappings
        self.function_name = functionName
        self.python_function_id = pythonFunctionId
        self.function_type = functionType
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        return f"PythonFunction(notebook_id={repr(self.notebook_id)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  function_variable_mappings={repr(self.function_variable_mappings)},\n  function_name={repr(self.function_name)},\n  python_function_id={repr(self.python_function_id)},\n  function_type={repr(self.function_type)},\n  code_source={repr(self.code_source)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'notebook_id': self.notebook_id, 'name': self.name, 'created_at': self.created_at, 'function_variable_mappings': self.function_variable_mappings, 'function_name': self.function_name, 'python_function_id': self.python_function_id, 'function_type': self.function_type, 'code_source': self._get_attribute_as_dict(self.code_source)}

    def add_graph_to_dashboard(self, graph_dashboard_id: str, function_variable_mappings: dict = None):
        """
        Add a python plot function to a dashboard

        Args:
            graph_dashboard_id (str): The graph dashboard id to update
            function_variable_mappings (dict): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].

        Returns:
            GraphDashboard: An object describing the graph dashboard
        """
        return self.client.add_graph_to_dashboard(self.python_function_id, graph_dashboard_id, function_variable_mappings)

    def update_graph_to_dashboard(self, graph_dashboard_id: str, function_variable_mappings: dict = None):
        """
        Update a python plot function to a dashboard

        Args:
            graph_dashboard_id (str): The graph dashboard id to update
            function_variable_mappings (dict): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].

        Returns:
            GraphDashboard: An object describing the graph dashboard
        """
        return self.client.update_graph_to_dashboard(self.python_function_id, graph_dashboard_id, function_variable_mappings)

    def describe_graph_for_dashboard(self, graph_dashboard_id: str):
        """
        Describes a python plot to a graph dashboard

        Args:
            graph_dashboard_id (str): The graph dashboard id to update

        Returns:
            PythonFunction: An object describing the graph dashboard
        """
        return self.client.describe_graph_for_dashboard(self.python_function_id, graph_dashboard_id)
