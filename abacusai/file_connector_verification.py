

class FileConnectorVerification():
    '''

    '''

    def __init__(self, client, verified=None, writePermission=None):
        self.client = client
        self.id = None
        self.verified = verified
        self.write_permission = writePermission

    def __repr__(self):
        return f"FileConnectorVerification(verified={repr(self.verified)}, write_permission={repr(self.write_permission)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'verified': self.verified, 'write_permission': self.write_permission}
