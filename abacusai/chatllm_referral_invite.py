from .return_class import AbstractApiClass


class ChatllmReferralInvite(AbstractApiClass):
    """
        The response of the Chatllm Referral Invite for different emails

        Args:
            client (ApiClient): An authenticated API Client instance
            userAlreadyExists (list): List of user emails not successfullt invited, because they are already registered users.
            successfulInvites (list): List of users successfully invited.
    """

    def __init__(self, client, userAlreadyExists=None, successfulInvites=None):
        super().__init__(client, None)
        self.user_already_exists = userAlreadyExists
        self.successful_invites = successfulInvites
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'user_already_exists': repr(
            self.user_already_exists), f'successful_invites': repr(self.successful_invites)}
        class_name = "ChatllmReferralInvite"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'user_already_exists': self.user_already_exists,
                'successful_invites': self.successful_invites}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
