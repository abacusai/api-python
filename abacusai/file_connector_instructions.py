from .return_class import AbstractApiClass


class FileConnectorInstructions(AbstractApiClass):
    """

    """

    def __init__(self, client, verified=None, writePermission=None, authOptions=None):
        super().__init__(client, None)
        self.verified = verified
        self.write_permission = writePermission
        self.auth_options = authOptions

    def __repr__(self):
        return f"FileConnectorInstructions(verified={repr(self.verified)},\n  write_permission={repr(self.write_permission)},\n  auth_options={repr(self.auth_options)})"

    def to_dict(self):
        return {'verified': self.verified, 'write_permission': self.write_permission, 'auth_options': self.auth_options}
