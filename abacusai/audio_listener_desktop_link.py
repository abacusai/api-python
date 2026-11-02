from .return_class import AbstractApiClass


class AudioListenerDesktopLink(AbstractApiClass):
    """
        The Listener desktop app's link, mailed to the signed-in user from the mobile app.

        Args:
            client (ApiClient): An authenticated API Client instance
            email (str): The address the link was sent to.
            sent (bool): Whether the mail went out.
    """

    def __init__(self, client, email=None, sent=None):
        super().__init__(client, None)
        self.email = email
        self.sent = sent
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'email': repr(self.email), f'sent': repr(self.sent)}
        class_name = "AudioListenerDesktopLink"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'email': self.email, 'sent': self.sent}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
