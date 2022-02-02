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
        return f"ApiKey(api_key_id={repr(self.api_key_id)},\n  api_key={repr(self.api_key)},\n  tag={repr(self.tag)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'api_key_id': self.api_key_id, 'api_key': self.api_key, 'tag': self.tag, 'created_at': self.created_at}

    def delete(self):
        """
        Delete a specified API Key. You can use the "listApiKeys" method to find the list of all API Key IDs.

        Args:
            api_key_id (str): The ID of the API key to delete.
        """
        return self.client.delete_api_key(self.api_key_id)
