

class ApplicationConnector():
    '''
        A connector to an external service
    '''

    def __init__(self, client, applicationConnectorId=None, service=None, name=None, createdAt=None, status=None, auth=None):
        self.client = client
        self.id = applicationConnectorId
        self.application_connector_id = applicationConnectorId
        self.service = service
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth

    def __repr__(self):
        return f"ApplicationConnector(application_connector_id={repr(self.application_connector_id)}, service={repr(self.service)}, name={repr(self.name)}, created_at={repr(self.created_at)}, status={repr(self.status)}, auth={repr(self.auth)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'application_connector_id': self.application_connector_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}

    def rename(self, name):
        return self.client.rename_application_connector(self.application_connector_id, name)

    def delete(self):
        return self.client.delete_application_connector(self.application_connector_id)

    def list_objects(self):
        return self.client.list_application_connector_objects(self.application_connector_id)

    def verify(self):
        return self.client.verify_application_connector(self.application_connector_id)
