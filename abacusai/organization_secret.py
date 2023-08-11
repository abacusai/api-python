from .return_class import AbstractApiClass


class OrganizationSecret(AbstractApiClass):
    """
        Organization secret

        Args:
            client (ApiClient): An authenticated API Client instance
            secretKey (str): The key of the secret
            value (str): The value of the secret
            createdAt (str): The date and time when the secret was created, in ISO-8601 format.
    """

    def __init__(self, client, secretKey=None, value=None, createdAt=None):
        super().__init__(client, None)
        self.secret_key = secretKey
        self.value = value
        self.created_at = createdAt

    def __repr__(self):
        return f"OrganizationSecret(secret_key={repr(self.secret_key)},\n  value={repr(self.value)},\n  created_at={repr(self.created_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'secret_key': self.secret_key, 'value': self.value, 'created_at': self.created_at}
