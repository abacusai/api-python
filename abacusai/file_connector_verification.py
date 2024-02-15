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
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'verified': repr(
            self.verified), f'write_permission': repr(self.write_permission)}
        class_name = "FileConnectorVerification"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'verified': self.verified,
                'write_permission': self.write_permission}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
