from .return_class import AbstractApiClass


class FileConnectorInstructions(AbstractApiClass):
    """
        An object with a full description of the cloud storage bucket authentication options and bucket policy. Returns an error message if the parameters are invalid.

        Args:
            client (ApiClient): An authenticated API Client instance
            verified (bool): `True` if the bucket has passed verification
            writePermission (bool): `True` if Abacus.AI has permission to write to this bucket
            authOptions (list[dict]): A list of options for giving Abacus.AI access to this bucket
    """

    def __init__(self, client, verified=None, writePermission=None, authOptions=None):
        super().__init__(client, None)
        self.verified = verified
        self.write_permission = writePermission
        self.auth_options = authOptions
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'verified': repr(self.verified), f'write_permission': repr(
            self.write_permission), f'auth_options': repr(self.auth_options)}
        class_name = "FileConnectorInstructions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'verified': self.verified, 'write_permission': self.write_permission,
                'auth_options': self.auth_options}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
