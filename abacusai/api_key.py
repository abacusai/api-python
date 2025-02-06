from .return_class import AbstractApiClass


class ApiKey(AbstractApiClass):
    """
        An API Key to authenticate requests to the Abacus.AI API

        Args:
            client (ApiClient): An authenticated API Client instance
            apiKeyId (str): The unique ID for the API key
            apiKey (str): The unique API key scoped to a specific organization. Value will be partially obscured.
            apiKeySuffix (str): The last 4 characters of the API key.
            tag (str): A user-friendly tag for the API key.
            type (str): The type of the API key, either 'default', 'code-llm', or 'computer-use'.
            createdAt (str): The timestamp when the API key was created.
            expiresAt (str): The timestamp when the API key will expire.
            isExpired (bool): Whether the API key has expired.
    """

    def __init__(self, client, apiKeyId=None, apiKey=None, apiKeySuffix=None, tag=None, type=None, createdAt=None, expiresAt=None, isExpired=None):
        super().__init__(client, apiKeyId)
        self.api_key_id = apiKeyId
        self.api_key = apiKey
        self.api_key_suffix = apiKeySuffix
        self.tag = tag
        self.type = type
        self.created_at = createdAt
        self.expires_at = expiresAt
        self.is_expired = isExpired
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'api_key_id': repr(self.api_key_id), f'api_key': repr(self.api_key), f'api_key_suffix': repr(self.api_key_suffix), f'tag': repr(
            self.tag), f'type': repr(self.type), f'created_at': repr(self.created_at), f'expires_at': repr(self.expires_at), f'is_expired': repr(self.is_expired)}
        class_name = "ApiKey"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'api_key_id': self.api_key_id, 'api_key': self.api_key, 'api_key_suffix': self.api_key_suffix, 'tag': self.tag,
                'type': self.type, 'created_at': self.created_at, 'expires_at': self.expires_at, 'is_expired': self.is_expired}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def delete(self):
        """
        Delete a specified API key.

        Args:
            api_key_id (str): The ID of the API key to delete.
        """
        return self.client.delete_api_key(self.api_key_id)
