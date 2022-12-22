from .return_class import AbstractApiClass


class GraphDashboard(AbstractApiClass):
    """
        A Graph Dashboard

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the model monitor.
            graphDashboardId (str): The unique identifier of the graph dashboard.
            createdAt (str): Date and time at which the graph dashboard was created.
            projectId (str): The project this model belongs to.
            pythonFunctionIds (list<unique string identifier>)): List of python graph ids included in the dashboard
            pythonFunctionNames (list<string>): List of names of each of the python functions
    """

    def __init__(self, client, name=None, graphDashboardId=None, createdAt=None, projectId=None, pythonFunctionIds=None, pythonFunctionNames=None):
        super().__init__(client, graphDashboardId)
        self.name = name
        self.graph_dashboard_id = graphDashboardId
        self.created_at = createdAt
        self.project_id = projectId
        self.python_function_ids = pythonFunctionIds
        self.python_function_names = pythonFunctionNames

    def __repr__(self):
        return f"GraphDashboard(name={repr(self.name)},\n  graph_dashboard_id={repr(self.graph_dashboard_id)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  python_function_ids={repr(self.python_function_ids)},\n  python_function_names={repr(self.python_function_names)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'graph_dashboard_id': self.graph_dashboard_id, 'created_at': self.created_at, 'project_id': self.project_id, 'python_function_ids': self.python_function_ids, 'python_function_names': self.python_function_names}

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            GraphDashboard: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a given graph dashboard.

        Args:
            graph_dashboard_id (str): The graph dashboard id

        Returns:
            GraphDashboard: An object describing the graph dashboard
        """
        return self.client.describe_graph_dashboard(self.graph_dashboard_id)

    def delete(self):
        """
        Deletes a graph dashboard

        Args:
            graph_dashboard_id (str): The graph dashboard id to delete
        """
        return self.client.delete_graph_dashboard(self.graph_dashboard_id)

    def update(self, name: str = None, python_function_ids: list = None):
        """
        Updates a graph dashboard

        Args:
            name (str): The name of the dashboard
            python_function_ids (list): The list of python function ids to use in the graph dashboard

        Returns:
            GraphDashboard: An object describing the graph dashboard
        """
        return self.client.update_graph_dashboard(self.graph_dashboard_id, name, python_function_ids)
