

class OrganizationGroup():
    '''
        An Organization Group. Defines the permissions available to the users who are members of the group.
    '''

    def __init__(self, client, organizationGroupId=None, permissions=None, groupName=None, defaultGroup=None, admin=None, createdAt=None):
        self.client = client
        self.id = organizationGroupId
        self.organization_group_id = organizationGroupId
        self.permissions = permissions
        self.group_name = groupName
        self.default_group = defaultGroup
        self.admin = admin
        self.created_at = createdAt

    def __repr__(self):
        return f"OrganizationGroup(organization_group_id={repr(self.organization_group_id)}, permissions={repr(self.permissions)}, group_name={repr(self.group_name)}, default_group={repr(self.default_group)}, admin={repr(self.admin)}, created_at={repr(self.created_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'organization_group_id': self.organization_group_id, 'permissions': self.permissions, 'group_name': self.group_name, 'default_group': self.default_group, 'admin': self.admin, 'created_at': self.created_at}

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_organization_group(self.organization_group_id)

    def add_permission(self, permission):
        return self.client.add_organization_group_permission(self.organization_group_id, permission)

    def remove_permission(self, permission):
        return self.client.remove_organization_group_permission(self.organization_group_id, permission)

    def delete(self):
        return self.client.delete_organization_group(self.organization_group_id)

    def add_user_to(self, email):
        return self.client.add_user_to_organization_group(self.organization_group_id, email)

    def remove_user_from(self, email):
        return self.client.remove_user_from_organization_group(self.organization_group_id, email)

    def set_default(self):
        return self.client.set_default_organization_group(self.organization_group_id)
