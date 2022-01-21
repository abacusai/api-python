from .return_class import AbstractApiClass


class FileConnectorVerification(AbstractApiClass):
    """
        To verify the file connector
    """

    def __init__(self, client, verified=None, writePermission=None):
        super().__init__(client, None)
        self.verified = verified
        self.write_permission = writePermission

    def __repr__(self):
        return f"FileConnectorVerification(verified={repr(self.verified)},\n  write_permission={repr(self.write_permission)})"

    def to_dict(self):
        return {'verified': self.verified, 'write_permission': self.write_permission}
