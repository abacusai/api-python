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
