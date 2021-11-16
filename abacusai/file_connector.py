from .return_class import AbstractApiClass


class FileConnector(AbstractApiClass):
    """
        Verification result for an external storage service
    """

    def __init__(self, client, bucket=None, verified=None, writePermission=None):
        super().__init__(client, None)
        self.bucket = bucket
        self.verified = verified
        self.write_permission = writePermission

    def __repr__(self):
        return f"FileConnector(bucket={repr(self.bucket)},\n  verified={repr(self.verified)},\n  write_permission={repr(self.write_permission)})"

    def to_dict(self):
        return {'bucket': self.bucket, 'verified': self.verified, 'write_permission': self.write_permission}
