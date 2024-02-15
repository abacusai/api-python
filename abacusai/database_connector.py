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
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'database_connector_id': repr(self.database_connector_id), f'service': repr(self.service), f'name': repr(
            self.name), f'status': repr(self.status), f'auth': repr(self.auth), f'created_at': repr(self.created_at)}
        class_name = "DatabaseConnector"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'database_connector_id': self.database_connector_id, 'service': self.service,
                'name': self.name, 'status': self.status, 'auth': self.auth, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def list_objects(self, fetch_raw_data: bool = False):
        """
        Lists querable objects in the database connector.

        Args:
            fetch_raw_data (bool): If true, return unfiltered objects.
        """
        return self.client.list_database_connector_objects(self.database_connector_id, fetch_raw_data)

    def get_object_schema(self, object_name: str = None, fetch_raw_data: bool = False):
        """
        Get the schema of an object in an database connector.

        Args:
            object_name (str): Unique identifier for the object in the external system.
            fetch_raw_data (bool): If true, return unfiltered list of columns.

        Returns:
            DatabaseConnectorSchema: The schema of the object.
        """
        return self.client.get_database_connector_object_schema(self.database_connector_id, object_name, fetch_raw_data)

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

    def query(self, query: str):
        """
        Runs a query in the specified database connector.

        Args:
            query (str): The query to be run in the database connector.
        """
        return self.client.query_database_connector(self.database_connector_id, query)
