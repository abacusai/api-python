from .return_class import AbstractApiClass


class EntriAuthToken(AbstractApiClass):
    """
        Entri Auth Token

        Args:
            client (ApiClient): An authenticated API Client instance
            token (str): The authentication token for Entri
            applicationId (str): application Id from Entri dashboard
            ttl (int): The duration in milliseconds for which the token is valid
    """

    def __init__(self, client, token=None, applicationId=None, ttl=None):
        super().__init__(client, None)
        self.token = token
        self.application_id = applicationId
        self.ttl = ttl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'token': repr(self.token), f'application_id': repr(
            self.application_id), f'ttl': repr(self.ttl)}
        class_name = "EntriAuthToken"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'token': self.token,
                'application_id': self.application_id, 'ttl': self.ttl}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
