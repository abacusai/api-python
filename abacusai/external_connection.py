

class ExternalConnection():
    '''

    '''

    def __init__(self, client, externalConnectionId=None, service=None, name=None, createdAt=None, status=None, auth=None):
        self.client = client
        self.id = externalConnectionId
        self.external_connection_id = externalConnectionId
        self.service = service
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth

    def __repr__(self):
        return f"ExternalConnection(external_connection_id={repr(self.external_connection_id)}, service={repr(self.service)}, name={repr(self.name)}, created_at={repr(self.created_at)}, status={repr(self.status)}, auth={repr(self.auth)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'external_connection_id': self.external_connection_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}

    def validate(self):
        return self.client.validate_external_connection(self.external_connection_id)
