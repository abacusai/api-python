from .return_class import AbstractApiClass


class UserSshKey(AbstractApiClass):
    """
        User SSH Key

        Args:
            client (ApiClient): An authenticated API Client instance
            userSshKeyId (id): The ID of the SSH key
            name (str): The name of the SSH key
            publicKey (str): The public key content
            keyType (str): The type of the SSH key
            fingerprint (str): The base64-encoded SHA-256 fingerprint
            createdAt (str): The creation timestamp
    """

    def __init__(self, client, userSshKeyId=None, name=None, publicKey=None, keyType=None, fingerprint=None, createdAt=None):
        super().__init__(client, userSshKeyId)
        self.user_ssh_key_id = userSshKeyId
        self.name = name
        self.public_key = publicKey
        self.key_type = keyType
        self.fingerprint = fingerprint
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'user_ssh_key_id': repr(self.user_ssh_key_id), f'name': repr(self.name), f'public_key': repr(
            self.public_key), f'key_type': repr(self.key_type), f'fingerprint': repr(self.fingerprint), f'created_at': repr(self.created_at)}
        class_name = "UserSshKey"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'user_ssh_key_id': self.user_ssh_key_id, 'name': self.name, 'public_key': self.public_key,
                'key_type': self.key_type, 'fingerprint': self.fingerprint, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
