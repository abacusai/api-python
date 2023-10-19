from .return_class import AbstractApiClass


class UserException(AbstractApiClass):
    """
        Exception information for errors in usercode.

        Args:
            client (ApiClient): An authenticated API Client instance
            type (str): The type of exception
            value (str): The value of the exception
            traceback (str): The traceback of the exception
    """

    def __init__(self, client, type=None, value=None, traceback=None):
        super().__init__(client, None)
        self.type = type
        self.value = value
        self.traceback = traceback

    def __repr__(self):
        repr_dict = {f'type': repr(self.type), f'value': repr(self.value), f'traceback': '[91m' + ((chr(39) * 3 + chr(
            10) + textwrap.indent(self.traceback, ' ' * 6) + chr(39) * 3) if self.traceback else str(self.traceback)) + '[0;0m'}
        class_name = "UserException"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'type': self.type, 'value': self.value,
                'traceback': self.traceback}
        return {key: value for key, value in resp.items() if value is not None}
