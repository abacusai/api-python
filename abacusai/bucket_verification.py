

class BucketVerification():
    '''

    '''

    def __init__(self, client, bucket=None, verified=None, writePermission=None, roleArn=None):
        self.client = client
        self.id = None
        self.bucket = bucket
        self.verified = verified
        self.write_permission = writePermission
        self.role_arn = roleArn

    def __repr__(self):
        return f"BucketVerification(bucket={repr(self.bucket)}, verified={repr(self.verified)}, write_permission={repr(self.write_permission)}, role_arn={repr(self.role_arn)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'bucket': self.bucket, 'verified': self.verified, 'write_permission': self.write_permission, 'role_arn': self.role_arn}
