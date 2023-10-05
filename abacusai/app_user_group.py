from .return_class import AbstractApiClass
from .user import User


class AppUserGroup(AbstractApiClass):
    """
        An app user group. This is used to determine which users have permissions for external chatbots.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the user group.
            userGroupId (str): The unique identifier of the user group.
            externalApplicationIds (list[str]): The ids of the external applications the group has access to.
            users (User): The users in the user group.
    """

    def __init__(self, client, name=None, userGroupId=None, externalApplicationIds=None, users={}):
        super().__init__(client, None)
        self.name = name
        self.user_group_id = userGroupId
        self.external_application_ids = externalApplicationIds
        self.users = client._build_class(User, users)

    def __repr__(self):
        return f"AppUserGroup(name={repr(self.name)},\n  user_group_id={repr(self.user_group_id)},\n  external_application_ids={repr(self.external_application_ids)},\n  users={repr(self.users)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'name': self.name, 'user_group_id': self.user_group_id, 'external_application_ids': self.external_application_ids, 'users': self._get_attribute_as_dict(self.users)}
