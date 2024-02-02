from .return_class import AbstractApiClass


class StreamingConnector(AbstractApiClass):
    """
        A connector to an external service

        Args:
            client (ApiClient): An authenticated API Client instance
            streamingConnectorId (str): The unique ID for the connection.
            service (str): The service this connection connects to
            name (str): A user-friendly name for the service
            createdAt (str): When the API key was created
            status (str): The status of the Database Connector
            auth (dict): Non-secret connection information for this connector
    """

    def __init__(self, client, streamingConnectorId=None, service=None, name=None, createdAt=None, status=None, auth=None):
        super().__init__(client, streamingConnectorId)
        self.streaming_connector_id = streamingConnectorId
        self.service = service
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth

    def __repr__(self):
        repr_dict = {f'streaming_connector_id': repr(self.streaming_connector_id), f'service': repr(self.service), f'name': repr(
            self.name), f'created_at': repr(self.created_at), f'status': repr(self.status), f'auth': repr(self.auth)}
        class_name = "StreamingConnector"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'streaming_connector_id': self.streaming_connector_id, 'service': self.service,
                'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}
        return {key: value for key, value in resp.items() if value is not None}

    def verify(self):
        """
        Checks to see if Abacus.AI can access the streaming connector.

        Args:
            streaming_connector_id (str): Unique string identifier for the streaming connector to be checked for Abacus.AI access.
        """
        return self.client.verify_streaming_connector(self.streaming_connector_id)

    def rename(self, name: str):
        """
        Renames a Streaming Connector

        Args:
            name (str): A new name for the streaming connector.
        """
        return self.client.rename_streaming_connector(self.streaming_connector_id, name)

    def delete(self):
        """
        Delete a streaming connector.

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector.
        """
        return self.client.delete_streaming_connector(self.streaming_connector_id)
