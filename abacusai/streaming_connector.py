

class StreamingConnector():
    '''
        A connector to an external service
    '''

    def __init__(self, client, streamingConnectorId=None, service=None, name=None, createdAt=None, status=None, auth=None):
        self.client = client
        self.id = streamingConnectorId
        self.streaming_connector_id = streamingConnectorId
        self.service = service
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth

    def __repr__(self):
        return f"StreamingConnector(streaming_connector_id={repr(self.streaming_connector_id)}, service={repr(self.service)}, name={repr(self.name)}, created_at={repr(self.created_at)}, status={repr(self.status)}, auth={repr(self.auth)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'streaming_connector_id': self.streaming_connector_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}
