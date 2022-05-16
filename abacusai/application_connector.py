from .return_class import AbstractApiClass


class ApplicationConnector(AbstractApiClass):
    """
        A connector to an external service

        Args:
            client (ApiClient): An authenticated API Client instance
            applicationConnectorId (str): The unique ID for the connection.
            service (str): The service this connection connects to
            name (str): A user-friendly name for the service
            createdAt (str): When the API key was created
            status (str): The status of the Application Connector
            auth (dict): Non-secret connection information for this connector
    """

    def __init__(self, client, applicationConnectorId=None, service=None, name=None, createdAt=None, status=None, auth=None):
        super().__init__(client, applicationConnectorId)
        self.application_connector_id = applicationConnectorId
        self.service = service
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth

    def __repr__(self):
        return f"ApplicationConnector(application_connector_id={repr(self.application_connector_id)},\n  service={repr(self.service)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  status={repr(self.status)},\n  auth={repr(self.auth)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'application_connector_id': self.application_connector_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}

    def create_feature_group_from_git(self, branch_name: str, table_name: str, function_name: str, module_name: str, python_root: str = None, input_feature_groups: list = None, description: str = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None):
        """
        Creates a new feature group from a ZIP file.

        Args:
            branch_name (str): Name of the branch in the git repository to be used for training.
            table_name (str): The unique name to be given to the feature group.
            function_name (str): Name of the function found in the module that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            description (str): The description about the feature group.
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            FeatureGroup: The created feature group
        """
        return self.client.create_feature_group_from_git(self.application_connector_id, branch_name, table_name, function_name, module_name, python_root, input_feature_groups, description, cpu_size, memory, package_requirements)

    def rename(self, name: str):
        """
        Renames an Application Connector

        Args:
            name (str): A new name for the application connector
        """
        return self.client.rename_application_connector(self.application_connector_id, name)

    def delete(self):
        """
        Delete a application connector.

        Args:
            application_connector_id (str): The unique identifier for the application connector.
        """
        return self.client.delete_application_connector(self.application_connector_id)

    def list_objects(self):
        """
        Lists querable objects in the application connector.

        Args:
            application_connector_id (str): The unique identifier for the application connector.
        """
        return self.client.list_application_connector_objects(self.application_connector_id)

    def verify(self):
        """
        Checks to see if Abacus.AI can access the Application.

        Args:
            application_connector_id (str): The unique identifier for the application connector.
        """
        return self.client.verify_application_connector(self.application_connector_id)
