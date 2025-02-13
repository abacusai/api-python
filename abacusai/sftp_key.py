from .return_class import AbstractApiClass


class SftpKey(AbstractApiClass):
    """
        An SFTP key

        Args:
            client (ApiClient): An authenticated API Client instance
            keyName (str): The name of the key
            publicKey (str): The public key
    """

    def __init__(self, client, keyName=None, publicKey=None):
        super().__init__(client, None)
        self.key_name = keyName
        self.public_key = publicKey
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'key_name': repr(
            self.key_name), f'public_key': repr(self.public_key)}
        class_name = "SftpKey"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'key_name': self.key_name, 'public_key': self.public_key}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
