from .organization_group import OrganizationGroup
from .return_class import AbstractApiClass


class User(AbstractApiClass):
    """
        A Abacus.AI User
    """

    def __init__(self, client, name=None, email=None, createdAt=None, status=None, organizationGroups={}):
        super().__init__(client, None)
        self.name = name
        self.email = email
        self.created_at = createdAt
        self.status = status
        self.organization_groups = client._build_class(
            OrganizationGroup, organizationGroups)

    def __repr__(self):
        return f"User(name={repr(self.name)},\n  email={repr(self.email)},\n  created_at={repr(self.created_at)},\n  status={repr(self.status)},\n  organization_groups={repr(self.organization_groups)})"

    def to_dict(self):
        return {'name': self.name, 'email': self.email, 'created_at': self.created_at, 'status': self.status, 'organization_groups': self._get_attribute_as_dict(self.organization_groups)}
