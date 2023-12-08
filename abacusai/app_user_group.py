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
            invitedUserEmails (list[str]): The emails of the users invited to the user group who have not yet accepted the invite.
            publicUserGroup (bool): Boolean flag whether the app user group is the public user group of the org or not
            users (User): The users in the user group.
    """

    def __init__(self, client, name=None, userGroupId=None, externalApplicationIds=None, invitedUserEmails=None, publicUserGroup=None, users={}):
        super().__init__(client, None)
        self.name = name
        self.user_group_id = userGroupId
        self.external_application_ids = externalApplicationIds
        self.invited_user_emails = invitedUserEmails
        self.public_user_group = publicUserGroup
        self.users = client._build_class(User, users)

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'user_group_id': repr(self.user_group_id), f'external_application_ids': repr(
            self.external_application_ids), f'invited_user_emails': repr(self.invited_user_emails), f'public_user_group': repr(self.public_user_group), f'users': repr(self.users)}
        class_name = "AppUserGroup"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'user_group_id': self.user_group_id, 'external_application_ids': self.external_application_ids,
                'invited_user_emails': self.invited_user_emails, 'public_user_group': self.public_user_group, 'users': self._get_attribute_as_dict(self.users)}
        return {key: value for key, value in resp.items() if value is not None}
