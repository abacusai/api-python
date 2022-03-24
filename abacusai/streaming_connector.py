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
            auth (dict): 
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
        return f"StreamingConnector(streaming_connector_id={repr(self.streaming_connector_id)},\n  service={repr(self.service)},\n  name={repr(self.name)},\n  created_at={repr(self.created_at)},\n  status={repr(self.status)},\n  auth={repr(self.auth)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'streaming_connector_id': self.streaming_connector_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}

    def verify(self):
        """
        Checks to see if Abacus.AI can access the streaming connector.

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector.
        """
        return self.client.verify_streaming_connector(self.streaming_connector_id)

    def rename(self, name: str):
        """
        Renames a Streaming Connector

        Args:
            name (str): A new name for the streaming connector
        """
        return self.client.rename_streaming_connector(self.streaming_connector_id, name)

    def delete(self):
        """
        Delete a streaming connector.

        Args:
            streaming_connector_id (str): The unique identifier for the streaming connector.
        """
        return self.client.delete_streaming_connector(self.streaming_connector_id)
