from .return_class import AbstractApiClass


class ApiKey(AbstractApiClass):
    """
        An API Key to authenticate requests to the Abacus.AI API

        Args:
            client (ApiClient): An authenticated API Client instance
            apiKeyId (str): The unique ID for the API key
            apiKey (str): The unique API key scoped to a specific organization. Value will be partially obscured.
            tag (str): A user-friendly tag for the API key.
            createdAt (str): The timestamp when the API key was created.
    """

    def __init__(self, client, apiKeyId=None, apiKey=None, tag=None, createdAt=None):
        super().__init__(client, apiKeyId)
        self.api_key_id = apiKeyId
        self.api_key = apiKey
        self.tag = tag
        self.created_at = createdAt

    def __repr__(self):
        repr_dict = {f'api_key_id': repr(self.api_key_id), f'api_key': repr(
            self.api_key), f'tag': repr(self.tag), f'created_at': repr(self.created_at)}
        class_name = "ApiKey"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'api_key_id': self.api_key_id, 'api_key': self.api_key,
                'tag': self.tag, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None}

    def delete(self):
        """
        Delete a specified API key.

        Args:
            api_key_id (str): The ID of the API key to delete.
        """
        return self.client.delete_api_key(self.api_key_id)
