from .return_class import AbstractApiClass


class ChatllmReferralInvite(AbstractApiClass):
    """
        The response of the Chatllm Referral Invite for different emails

        Args:
            client (ApiClient): An authenticated API Client instance
            userAlreadyExists (list): List of user emails not successfullt invited, because they are already registered users.
            successfulInvites (list): List of users successfully invited.
            failedToSend (list): Emails whose invite could not be sent from the user's Gmail account.
    """

    def __init__(self, client, userAlreadyExists=None, successfulInvites=None, failedToSend=None):
        super().__init__(client, None)
        self.user_already_exists = userAlreadyExists
        self.successful_invites = successfulInvites
        self.failed_to_send = failedToSend
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'user_already_exists': repr(self.user_already_exists), f'successful_invites': repr(
            self.successful_invites), f'failed_to_send': repr(self.failed_to_send)}
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
                'successful_invites': self.successful_invites, 'failed_to_send': self.failed_to_send}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
