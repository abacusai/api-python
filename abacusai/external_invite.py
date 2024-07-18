from .return_class import AbstractApiClass


class ExternalInvite(AbstractApiClass):
    """
        The response of the invites for different emails

        Args:
            client (ApiClient): An authenticated API Client instance
            userAlreadyInOrg (list): List of user emails not successfully invited, because they are already in the organization.
            userAlreadyInAppGroup (list): List of user emails not successfully invited, because they are already in the application group.
            userExistsAsInternal (list): List of user emails not successfully invited, because they are already internal users.
            successfulInvites (list): List of users successfully invited.
    """

    def __init__(self, client, userAlreadyInOrg=None, userAlreadyInAppGroup=None, userExistsAsInternal=None, successfulInvites=None):
        super().__init__(client, None)
        self.user_already_in_org = userAlreadyInOrg
        self.user_already_in_app_group = userAlreadyInAppGroup
        self.user_exists_as_internal = userExistsAsInternal
        self.successful_invites = successfulInvites
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'user_already_in_org': repr(self.user_already_in_org), f'user_already_in_app_group': repr(
            self.user_already_in_app_group), f'user_exists_as_internal': repr(self.user_exists_as_internal), f'successful_invites': repr(self.successful_invites)}
        class_name = "ExternalInvite"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'user_already_in_org': self.user_already_in_org, 'user_already_in_app_group': self.user_already_in_app_group,
                'user_exists_as_internal': self.user_exists_as_internal, 'successful_invites': self.successful_invites}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
