from .return_class import AbstractApiClass


class FileConnectorVerification(AbstractApiClass):
    """
        To verify the file connector

        Args:
            client (ApiClient): An authenticated API Client instance
            verified (bool): `true` if the bucket has passed verification
            writePermission (bool): `true` if Abacus.AI has permission to write to this bucket
    """

    def __init__(self, client, verified=None, writePermission=None):
        super().__init__(client, None)
        self.verified = verified
        self.write_permission = writePermission

    def __repr__(self):
        return f"FileConnectorVerification(verified={repr(self.verified)},\n  write_permission={repr(self.write_permission)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'verified': self.verified, 'write_permission': self.write_permission}
