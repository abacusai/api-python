from .return_class import AbstractApiClass


class HostedAppFileRead(AbstractApiClass):
    """
        Result of reading file content from a hosted app container.

        Args:
            client (ApiClient): An authenticated API Client instance
            content (str): The contents of the file or a portion of it.
            start (int): If present, the starting position of the read.
            end (int): If present, the last position in the file returned in this read.
            retcode (int): If the read is associated with a log the return code of the command.
    """

    def __init__(self, client, content=None, start=None, end=None, retcode=None):
        super().__init__(client, None)
        self.content = content
        self.start = start
        self.end = end
        self.retcode = retcode
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'content': repr(self.content), f'start': repr(
            self.start), f'end': repr(self.end), f'retcode': repr(self.retcode)}
        class_name = "HostedAppFileRead"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'content': self.content, 'start': self.start,
                'end': self.end, 'retcode': self.retcode}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
