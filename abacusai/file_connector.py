from .return_class import AbstractApiClass


class FileConnector(AbstractApiClass):
    """
        Verification result for an external storage service

        Args:
            client (ApiClient): An authenticated API Client instance
            bucket (str): The address of the bucket. eg., `s3://your-bucket`
            verified (bool): `true` if the bucket has passed verification
            writePermission (bool): `true` if Abacus.AI has permission to write to this bucket
            authExpiresAt (str): The time when the file connector's auth expires, if applicable
    """

    def __init__(self, client, bucket=None, verified=None, writePermission=None, authExpiresAt=None):
        super().__init__(client, None)
        self.bucket = bucket
        self.verified = verified
        self.write_permission = writePermission
        self.auth_expires_at = authExpiresAt

    def __repr__(self):
        repr_dict = {f'bucket': repr(self.bucket), f'verified': repr(self.verified), f'write_permission': repr(
            self.write_permission), f'auth_expires_at': repr(self.auth_expires_at)}
        class_name = "FileConnector"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'bucket': self.bucket, 'verified': self.verified,
                'write_permission': self.write_permission, 'auth_expires_at': self.auth_expires_at}
        return {key: value for key, value in resp.items() if value is not None}
