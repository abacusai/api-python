

class DatabaseConnector():
    '''

    '''

    def __init__(self, client, databaseConnectorId=None, service=None, name=None, createdAt=None, status=None, auth=None):
        self.client = client
        self.id = databaseConnectorId
        self.database_connector_id = databaseConnectorId
        self.service = service
        self.name = name
        self.created_at = createdAt
        self.status = status
        self.auth = auth

    def __repr__(self):
        return f"DatabaseConnector(database_connector_id={repr(self.database_connector_id)}, service={repr(self.service)}, name={repr(self.name)}, created_at={repr(self.created_at)}, status={repr(self.status)}, auth={repr(self.auth)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'database_connector_id': self.database_connector_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}

    def list_objects(self):
        return self.client.list_database_connector_objects(self.database_connector_id)

    def get_object_schema(self, object_name=None):
        return self.client.get_database_connector_object_schema(self.database_connector_id, object_name)

    def verify(self):
        return self.client.verify_database_connector(self.database_connector_id)

    def remove(self):
        return self.client.remove_database_connector(self.database_connector_id)
