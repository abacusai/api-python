from .return_class import AbstractApiClass


class UnifiedConnector(AbstractApiClass):
    """
        A unified connector that can handle both application and database connectors.

        Args:
            client (ApiClient): An authenticated API Client instance
            applicationConnectorId (str): The unique ID for the connection.
            databaseConnectorId (str): The unique ID for the connection.
            service (str): The service this connection connects to
            name (str): A user-friendly name for the service
            createdAt (str): When the API key was created
            status (str): The status of the Application Connector
            auth (dict): Non-secret connection information for this connector
    """

    def __init__(self, client, applicationConnectorId=None, databaseConnectorId=None, service=None, name=None, createdAt=None, status=None, auth=None):
        super().__init__(client, None)
        self.application_connector_id = applicationConnectorId
        self.database_connector_id = databaseConnectorId
        self.service = service
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'application_connector_id': repr(self.application_connector_id), f'database_connector_id': repr(self.database_connector_id), f'service': repr(
            self.service), f'name': repr(self.name), f'created_at': repr(self.created_at), f'status': repr(self.status), f'auth': repr(self.auth)}
        class_name = "UnifiedConnector"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'application_connector_id': self.application_connector_id, 'database_connector_id': self.database_connector_id,
                'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
