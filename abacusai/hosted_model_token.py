from .return_class import AbstractApiClass


class HostedModelToken(AbstractApiClass):
    """
        A hosted model authentication token that is used to authenticate requests to an abacus hosted model

        Args:
            client (ApiClient): An authenticated API Client instance
            createdAt (str): When the token was created
            tag (str): A user-friendly tag for the API key.
            trailingAuthToken (str): The last four characters of the un-encrypted auth token
            hostedModelTokenId (str): The unique identifier attached to this authenticaion token
    """

    def __init__(self, client, createdAt=None, tag=None, trailingAuthToken=None, hostedModelTokenId=None):
        super().__init__(client, hostedModelTokenId)
        self.created_at = createdAt
        self.tag = tag
        self.trailing_auth_token = trailingAuthToken
        self.hosted_model_token_id = hostedModelTokenId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'created_at': repr(self.created_at), f'tag': repr(self.tag), f'trailing_auth_token': repr(
            self.trailing_auth_token), f'hosted_model_token_id': repr(self.hosted_model_token_id)}
        class_name = "HostedModelToken"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'created_at': self.created_at, 'tag': self.tag, 'trailing_auth_token':
                self.trailing_auth_token, 'hosted_model_token_id': self.hosted_model_token_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
