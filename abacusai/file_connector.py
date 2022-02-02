from .return_class import AbstractApiClass


class FileConnector(AbstractApiClass):
    """
        Verification result for an external storage service

        Args:
            client (ApiClient): An authenticated API Client instance
            bucket (str): The address of the bucket. eg., `s3://your-bucket`
            verified (bool): `true` if the bucket has passed verification
            writePermission (bool): `true` if Abacus.AI has permission to write to this bucket
    """

    def __init__(self, client, bucket=None, verified=None, writePermission=None):
        super().__init__(client, None)
        self.bucket = bucket
        self.verified = verified
        self.write_permission = writePermission

    def __repr__(self):
        return f"FileConnector(bucket={repr(self.bucket)},\n  verified={repr(self.verified)},\n  write_permission={repr(self.write_permission)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'bucket': self.bucket, 'verified': self.verified, 'write_permission': self.write_permission}
