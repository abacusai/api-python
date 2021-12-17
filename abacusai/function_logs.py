import textwrap

from .return_class import AbstractApiClass
from .user_exception import UserException


class FunctionLogs(AbstractApiClass):
    """
        Logs from an invocation of a function.
    """

    def __init__(self, client, function=None, stats=None, stdout=None, stderr=None, exception={}):
        super().__init__(client, None)
        self.function = function
        self.stats = stats
        self.stdout = stdout
        self.stderr = stderr
        self.exception = client._build_class(UserException, exception)

    def __repr__(self):
        return f"FunctionLogs(function={repr(self.function)},\n  stats={repr(self.stats)},\n  stdout={'[92m' + ((chr(39) * 3 + chr(10) + textwrap.indent(self.stdout, ' ' * 6) + chr(39) * 3) if self.stdout else str(self.stdout)) + '[0;0m'},\n  stderr={'[91m' + ((chr(39) * 3 + chr(10) + textwrap.indent(self.stderr, ' ' * 6) + chr(39) * 3) if self.stderr else str(self.stderr)) + '[0;0m'},\n  exception={repr(self.exception)})"

    def to_dict(self):
        return {'function': self.function, 'stats': self.stats, 'stdout': self.stdout, 'stderr': self.stderr, 'exception': self._get_attribute_as_dict(self.exception)}
