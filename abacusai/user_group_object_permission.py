from .return_class import AbstractApiClass


class UserGroupObjectPermission(AbstractApiClass):
    """
        A user group object permission

        Args:
            client (ApiClient): An authenticated API Client instance
            userGroupId (str): The unique identifier of the user group.
            userGroupName (str): The name of the user group.
            objectId (str): The unique identifier of the object.
            permission (str): The permission level (e.g., 'ALL').
    """

    def __init__(self, client, userGroupId=None, userGroupName=None, objectId=None, permission=None):
        super().__init__(client, None)
        self.user_group_id = userGroupId
        self.user_group_name = userGroupName
        self.object_id = objectId
        self.permission = permission
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'user_group_id': repr(self.user_group_id), f'user_group_name': repr(
            self.user_group_name), f'object_id': repr(self.object_id), f'permission': repr(self.permission)}
        class_name = "UserGroupObjectPermission"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'user_group_id': self.user_group_id, 'user_group_name': self.user_group_name,
                'object_id': self.object_id, 'permission': self.permission}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
