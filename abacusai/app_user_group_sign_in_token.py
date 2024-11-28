from .return_class import AbstractApiClass


class AppUserGroupSignInToken(AbstractApiClass):
    """
        User Group Sign In Token

        Args:
            client (ApiClient): An authenticated API Client instance
            token (str): The token to sign in the user
    """

    def __init__(self, client, token=None):
        super().__init__(client, None)
        self.token = token
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'token': repr(self.token)}
        class_name = "AppUserGroupSignInToken"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'token': self.token}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
