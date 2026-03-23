from .return_class import AbstractApiClass


class OrganizationSecret(AbstractApiClass):
    """
        Organization secret

        Args:
            client (ApiClient): An authenticated API Client instance
            secretKey (str): The key of the secret
            value (str): The value of the secret
            createdAt (str): The date and time when the secret was created, in ISO-8601 format.
            metadata (dict): Additional metadata for the secret (e.g., web_app_project_ids).
            createdByEmail (str): The email of the user who created the secret.
    """

    def __init__(self, client, secretKey=None, value=None, createdAt=None, metadata=None, createdByEmail=None):
        super().__init__(client, None)
        self.secret_key = secretKey
        self.value = value
        self.created_at = createdAt
        self.metadata = metadata
        self.created_by_email = createdByEmail
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'secret_key': repr(self.secret_key), f'value': repr(self.value), f'created_at': repr(
            self.created_at), f'metadata': repr(self.metadata), f'created_by_email': repr(self.created_by_email)}
        class_name = "OrganizationSecret"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'secret_key': self.secret_key, 'value': self.value, 'created_at': self.created_at,
                'metadata': self.metadata, 'created_by_email': self.created_by_email}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
