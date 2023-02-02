from .return_class import AbstractApiClass


class DatabaseConnector(AbstractApiClass):
    """
        A connector to an external service

        Args:
            client (ApiClient): An authenticated API Client instance
            databaseConnectorId (str): A unique string identifier for the connection.
            service (str): An enum string indicating the service this connection connects to.
            name (str): A user-friendly name for the service.
            status (str): The status of the database connector.
            auth (dict): Non-secret connection information for this connector.
            createdAt (str): The ISO-8601 string indicating when the API key was created.
    """

    def __init__(self, client, databaseConnectorId=None, service=None, name=None, status=None, auth=None, createdAt=None):
        super().__init__(client, databaseConnectorId)
        self.database_connector_id = databaseConnectorId
        self.service = service
        self.name = name
        self.status = status
        self.auth = auth
        self.created_at = createdAt

    def __repr__(self):
        return f"DatabaseConnector(database_connector_id={repr(self.database_connector_id)},\n  service={repr(self.service)},\n  name={repr(self.name)},\n  status={repr(self.status)},\n  auth={repr(self.auth)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'database_connector_id': self.database_connector_id, 'service': self.service, 'name': self.name, 'status': self.status, 'auth': self.auth, 'created_at': self.created_at}

    def list_objects(self):
        """
        Lists querable objects in the database connector.

        Args:
            database_connector_id (str): Unique string identifier for the database connector.
        """
        return self.client.list_database_connector_objects(self.database_connector_id)

    def get_object_schema(self, object_name: str = None):
        """
        Get the schema of an object in an database connector.

        Args:
            object_name (str): Unique identifier for the object in the external system.
        """
        return self.client.get_database_connector_object_schema(self.database_connector_id, object_name)

    def rename(self, name: str):
        """
        Renames a Database Connector

        Args:
            name (str): The new name for the Database Connector.
        """
        return self.client.rename_database_connector(self.database_connector_id, name)

    def verify(self):
        """
        Checks if Abacus.AI can access the specified database.

        Args:
            database_connector_id (str): Unique string identifier for the database connector.
        """
        return self.client.verify_database_connector(self.database_connector_id)

    def delete(self):
        """
        Delete a database connector.

        Args:
            database_connector_id (str): The unique identifier for the database connector.
        """
        return self.client.delete_database_connector(self.database_connector_id)
