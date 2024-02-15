from .organization_group import OrganizationGroup
from .return_class import AbstractApiClass


class User(AbstractApiClass):
    """
        An Abacus.AI User

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The User's name.
            email (str): The User's primary email address.
            createdAt (str): The date and time when the user joined Abacus.AI.
            status (str): `ACTIVE` when the user has accepted an invite to join the organization, else `INVITED`.
            organizationGroups (OrganizationGroup): List of Organization Groups this user belongs to.
    """

    def __init__(self, client, name=None, email=None, createdAt=None, status=None, organizationGroups={}):
        super().__init__(client, None)
        self.name = name
        self.email = email
        self.created_at = createdAt
        self.status = status
        self.organization_groups = client._build_class(
            OrganizationGroup, organizationGroups)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'email': repr(self.email), f'created_at': repr(
            self.created_at), f'status': repr(self.status), f'organization_groups': repr(self.organization_groups)}
        class_name = "User"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'email': self.email, 'created_at': self.created_at, 'status': self.status,
                'organization_groups': self._get_attribute_as_dict(self.organization_groups)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
