from .return_class import AbstractApiClass


class GraphDashboard(AbstractApiClass):
    """
        A Graph Dashboard

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The user-friendly name for the graph dashboard.
            graphDashboardId (str): The unique identifier of the graph dashboard.
            createdAt (str): Date and time at which the graph dashboard was created, in ISO-8601 format.
            projectId (str): The unique identifier of the project this graph dashboard belongs to.
            pythonFunctionIds (list[str]): List of Python function IDs included in the dashboard.
            plotReferenceIds (list[str]): List of the graph reference IDs for the plots to the dashboard.
            pythonFunctionNames (list[str]): List of names of each of the plots to the dashboard.
            projectName (str): The name the graph dashboard belongs to.
    """

    def __init__(self, client, name=None, graphDashboardId=None, createdAt=None, projectId=None, pythonFunctionIds=None, plotReferenceIds=None, pythonFunctionNames=None, projectName=None):
        super().__init__(client, graphDashboardId)
        self.name = name
        self.graph_dashboard_id = graphDashboardId
        self.created_at = createdAt
        self.project_id = projectId
        self.python_function_ids = pythonFunctionIds
        self.plot_reference_ids = plotReferenceIds
        self.python_function_names = pythonFunctionNames
        self.project_name = projectName

    def __repr__(self):
        return f"GraphDashboard(name={repr(self.name)},\n  graph_dashboard_id={repr(self.graph_dashboard_id)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  python_function_ids={repr(self.python_function_ids)},\n  plot_reference_ids={repr(self.plot_reference_ids)},\n  python_function_names={repr(self.python_function_names)},\n  project_name={repr(self.project_name)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'graph_dashboard_id': self.graph_dashboard_id, 'created_at': self.created_at, 'project_id': self.project_id, 'python_function_ids': self.python_function_ids, 'plot_reference_ids': self.plot_reference_ids, 'python_function_names': self.python_function_names, 'project_name': self.project_name}

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
            graph_dashboard_id (str): Unique identifier for the graph dashboard.

        Returns:
            GraphDashboard: An object containing information about the graph dashboard.
        """
        return self.client.describe_graph_dashboard(self.graph_dashboard_id)

    def delete(self):
        """
        Deletes a graph dashboard

        Args:
            graph_dashboard_id (str): Unique string identifier for the graph dashboard to be deleted.
        """
        return self.client.delete_graph_dashboard(self.graph_dashboard_id)

    def update(self, name: str = None, python_function_ids: list = None):
        """
        Updates a graph dashboard

        Args:
            name (str): Name of the dashboard.
            python_function_ids (list): List of unique string identifiers for the Python functions to be used in the graph dashboard.

        Returns:
            GraphDashboard: An object describing the graph dashboard.
        """
        return self.client.update_graph_dashboard(self.graph_dashboard_id, name, python_function_ids)
