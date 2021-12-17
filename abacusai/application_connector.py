from .return_class import AbstractApiClass


class ApplicationConnector(AbstractApiClass):
    """
        A connector to an external service
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
        return {'application_connector_id': self.application_connector_id, 'service': self.service, 'name': self.name, 'created_at': self.created_at, 'status': self.status, 'auth': self.auth}

    def rename(self, name):
        """Renames an Application Connector"""
        return self.client.rename_application_connector(self.application_connector_id, name)

    def delete(self):
        """Delete a application connector."""
        return self.client.delete_application_connector(self.application_connector_id)

    def list_objects(self):
        """Lists querable objects in the application connector."""
        return self.client.list_application_connector_objects(self.application_connector_id)

    def verify(self):
        """Checks to see if Abacus.AI can access the Application."""
        return self.client.verify_application_connector(self.application_connector_id)
