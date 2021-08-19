from .organization_group import OrganizationGroup


class User():
    '''
        A Abacus.AI User
    '''

    def __init__(self, client, name=None, email=None, createdAt=None, status=None, organizationGroups={}):
        self.client = client
        self.id = None
        self.name = name
        self.email = email
        self.created_at = createdAt
        self.status = status
        self.organization_groups = client._build_class(
            OrganizationGroup, organizationGroups)

    def __repr__(self):
        return f"User(name={repr(self.name)}, email={repr(self.email)}, created_at={repr(self.created_at)}, status={repr(self.status)}, organization_groups={repr(self.organization_groups)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'email': self.email, 'created_at': self.created_at, 'status': self.status, 'organization_groups': self.organization_groups.to_dict() if self.organization_groups else None}
