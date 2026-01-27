from .return_class import AbstractApiClass


class ApplicationConnector(AbstractApiClass):
    """
        A connector to an external service

        Args:
            client (ApiClient): An authenticated API Client instance
            applicationConnectorId (str): The unique ID for the connection.
            service (str): The service this connection connects to
            serviceName (str): For OAuth services, the specific provider name (e.g., 'spotify')
            name (str): A user-friendly name for the service
            createdAt (str): When the API key was created
            status (str): The status of the Application Connector
            auth (dict): Non-secret connection information for this connector
            baseConnector (str): The hashed ID of the config connector this user connector is based on (for config connector flows)
    """

    def __init__(self, client, applicationConnectorId=None, service=None, serviceName=None, name=None, createdAt=None, status=None, auth=None, baseConnector=None):
        super().__init__(client, applicationConnectorId)
        self.application_connector_id = applicationConnectorId
        self.service = service
        self.service_name = serviceName
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth
        self.base_connector = baseConnector
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'application_connector_id': repr(self.application_connector_id), f'service': repr(self.service), f'service_name': repr(self.service_name), f'name': repr(
            self.name), f'created_at': repr(self.created_at), f'status': repr(self.status), f'auth': repr(self.auth), f'base_connector': repr(self.base_connector)}
        class_name = "ApplicationConnector"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'application_connector_id': self.application_connector_id, 'service': self.service, 'service_name': self.service_name,
                'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth, 'base_connector': self.base_connector}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def rename(self, name: str):
        """
        Renames a Application Connector

        Args:
            name (str): A new name for the application connector.
        """
        return self.client.rename_application_connector(self.application_connector_id, name)

    def delete(self):
        """
        Delete an application connector.

        Args:
            application_connector_id (str): The unique identifier for the application connector.
        """
        return self.client.delete_application_connector(self.application_connector_id)

    def list_objects(self):
        """
        Lists querable objects in the application connector.

        Args:
            application_connector_id (str): Unique string identifier for the application connector.
        """
        return self.client.list_application_connector_objects(self.application_connector_id)

    def verify(self):
        """
        Checks if Abacus.AI can access the application using the provided application connector ID.

        Args:
            application_connector_id (str): Unique string identifier for the application connector.
        """
        return self.client.verify_application_connector(self.application_connector_id)
