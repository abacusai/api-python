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
            publicUserGroup (bool): Boolean flag whether the app user group is the public user group of the org or not.
            hasExternalApplicationReporting (bool): Whether users in the App User Group have permission to view all reports in their organization.
            isExternalServiceGroup (bool): Whether the App User Group corresponds to a user group that's defined in an external service (i.e Microsft Active Directory or Okta) or not
            externalServiceGroupId (str): The identifier that corresponds to the app user group's external service group representation
            users (User): The users in the user group.
    """

    def __init__(self, client, name=None, userGroupId=None, externalApplicationIds=None, invitedUserEmails=None, publicUserGroup=None, hasExternalApplicationReporting=None, isExternalServiceGroup=None, externalServiceGroupId=None, users={}):
        super().__init__(client, None)
        self.name = name
        self.user_group_id = userGroupId
        self.external_application_ids = externalApplicationIds
        self.invited_user_emails = invitedUserEmails
        self.public_user_group = publicUserGroup
        self.has_external_application_reporting = hasExternalApplicationReporting
        self.is_external_service_group = isExternalServiceGroup
        self.external_service_group_id = externalServiceGroupId
        self.users = client._build_class(User, users)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'user_group_id': repr(self.user_group_id), f'external_application_ids': repr(self.external_application_ids), f'invited_user_emails': repr(self.invited_user_emails), f'public_user_group': repr(
            self.public_user_group), f'has_external_application_reporting': repr(self.has_external_application_reporting), f'is_external_service_group': repr(self.is_external_service_group), f'external_service_group_id': repr(self.external_service_group_id), f'users': repr(self.users)}
        class_name = "AppUserGroup"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'user_group_id': self.user_group_id, 'external_application_ids': self.external_application_ids, 'invited_user_emails': self.invited_user_emails, 'public_user_group': self.public_user_group,
                'has_external_application_reporting': self.has_external_application_reporting, 'is_external_service_group': self.is_external_service_group, 'external_service_group_id': self.external_service_group_id, 'users': self._get_attribute_as_dict(self.users)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
