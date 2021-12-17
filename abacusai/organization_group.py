from .return_class import AbstractApiClass


class OrganizationGroup(AbstractApiClass):
    """
        An Organization Group. Defines the permissions available to the users who are members of the group.
    """

    def __init__(self, client, organizationGroupId=None, permissions=None, groupName=None, defaultGroup=None, admin=None, createdAt=None):
        super().__init__(client, organizationGroupId)
        self.organization_group_id = organizationGroupId
        self.permissions = permissions
        self.group_name = groupName
        self.default_group = defaultGroup
        self.admin = admin
        self.created_at = createdAt

    def __repr__(self):
        return f"OrganizationGroup(organization_group_id={repr(self.organization_group_id)},\n  permissions={repr(self.permissions)},\n  group_name={repr(self.group_name)},\n  default_group={repr(self.default_group)},\n  admin={repr(self.admin)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        return {'organization_group_id': self.organization_group_id, 'permissions': self.permissions, 'group_name': self.group_name, 'default_group': self.default_group, 'admin': self.admin, 'created_at': self.created_at}

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Returns the specific organization group passes in by the user."""
        return self.client.describe_organization_group(self.organization_group_id)

    def add_permission(self, permission):
        """Adds a permission to the specified Organization Group"""
        return self.client.add_organization_group_permission(self.organization_group_id, permission)

    def remove_permission(self, permission):
        """Removes a permission from the specified Organization Group"""
        return self.client.remove_organization_group_permission(self.organization_group_id, permission)

    def delete(self):
        """Deletes the specified Organization Group from the organization."""
        return self.client.delete_organization_group(self.organization_group_id)

    def add_user_to(self, email):
        """Adds a user to the specified Organization Group"""
        return self.client.add_user_to_organization_group(self.organization_group_id, email)

    def remove_user_from(self, email):
        """Removes a user from an Organization Group"""
        return self.client.remove_user_from_organization_group(self.organization_group_id, email)

    def set_default(self):
        """Sets the default Organization Group that all new users that join an organization are automatically added to"""
        return self.client.set_default_organization_group(self.organization_group_id)
