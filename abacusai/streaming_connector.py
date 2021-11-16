from .return_class import AbstractApiClass


class StreamingConnector(AbstractApiClass):
    """
        A connector to an external service
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
        return {'streaming_connector_id': self.streaming_connector_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}
