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

    def __repr__(self):
        return f"FileConnectorInstructions(verified={repr(self.verified)},\n  write_permission={repr(self.write_permission)},\n  auth_options={repr(self.auth_options)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'verified': self.verified, 'write_permission': self.write_permission, 'auth_options': self.auth_options}
