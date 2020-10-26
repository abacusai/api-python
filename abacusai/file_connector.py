

class FileConnector():
    '''

    '''

    def __init__(self, client, bucket=None, verified=None, writePermission=None):
        self.client = client
        self.id = None
        self.bucket = bucket
        self.verified = verified
        self.write_permission = writePermission

    def __repr__(self):
        return f"FileConnector(bucket={repr(self.bucket)}, verified={repr(self.verified)}, write_permission={repr(self.write_permission)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'bucket': self.bucket, 'verified': self.verified, 'write_permission': self.write_permission}
