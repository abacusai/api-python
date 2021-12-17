import textwrap

from .return_class import AbstractApiClass


class UserException(AbstractApiClass):
    """
        Exception information for errors in usercode.
    """

    def __init__(self, client, type=None, value=None, traceback=None):
        super().__init__(client, None)
        self.type = type
        self.value = value
        self.traceback = traceback

    def __repr__(self):
        return f"UserException(type={repr(self.type)},\n  value={repr(self.value)},\n  traceback={'[91m' + ((chr(39) * 3 + chr(10) + textwrap.indent(self.traceback, ' ' * 6) + chr(39) * 3) if self.traceback else str(self.traceback)) + '[0;0m'})"

    def to_dict(self):
        return {'type': self.type, 'value': self.value, 'traceback': self.traceback}
