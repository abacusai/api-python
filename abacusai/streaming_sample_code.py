from .return_class import AbstractApiClass


class StreamingSampleCode(AbstractApiClass):
    """
        Sample code for adding to a streaming feature group with examples from different locations.

        Args:
            client (ApiClient): An authenticated API Client instance
            python (str): The python code sample.
            curl (str): The curl code sample.
            console (str): The console code sample
    """

    def __init__(self, client, python=None, curl=None, console=None):
        super().__init__(client, None)
        self.python = python
        self.curl = curl
        self.console = console

    def __repr__(self):
        repr_dict = {f'python': repr(self.python), f'curl': repr(
            self.curl), f'console': repr(self.console)}
        class_name = "StreamingSampleCode"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'python': self.python,
                'curl': self.curl, 'console': self.console}
        return {key: value for key, value in resp.items() if value is not None}
