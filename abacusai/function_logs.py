from .return_class import AbstractApiClass
from .user_exception import UserException


class FunctionLogs(AbstractApiClass):
    """
        Logs from an invocation of a function.

        Args:
            client (ApiClient): An authenticated API Client instance
            function (str): The function this is logging
            stats (dict): Statistics for the start and end time execution for this function
            stdout (str): Standard out logs
            stderr (str): Standard error logs
            algorithm (str): Algorithm name for this function
            exception (UserException): The exception stacktrace
    """

    def __init__(self, client, function=None, stats=None, stdout=None, stderr=None, algorithm=None, exception={}):
        super().__init__(client, None)
        self.function = function
        self.stats = stats
        self.stdout = stdout
        self.stderr = stderr
        self.algorithm = algorithm
        self.exception = client._build_class(UserException, exception)

    def __repr__(self):
        repr_dict = {f'function': repr(self.function), f'stats': repr(self.stats), f'stdout': '[92m' + ((chr(39) * 3 + chr(10) + textwrap.indent(self.stdout, ' ' * 6) + chr(39) * 3) if self.stdout else str(self.stdout)) + '[0;0m', f'stderr': '[91m' + (
            (chr(39) * 3 + chr(10) + textwrap.indent(self.stderr, ' ' * 6) + chr(39) * 3) if self.stderr else str(self.stderr)) + '[0;0m', f'algorithm': repr(self.algorithm), f'exception': repr(self.exception)}
        class_name = "FunctionLogs"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'function': self.function, 'stats': self.stats, 'stdout': self.stdout, 'stderr': self.stderr,
                'algorithm': self.algorithm, 'exception': self._get_attribute_as_dict(self.exception)}
        return {key: value for key, value in resp.items() if value is not None}
