from .return_class import AbstractApiClass


class SessionTranscript(AbstractApiClass):
    """
        A session transcript

        Args:
            client (ApiClient): An authenticated API Client instance
            role (str): The role of the transcript.
            content (str): The content of the transcript.
    """

    def __init__(self, client, role=None, content=None):
        super().__init__(client, None)
        self.role = role
        self.content = content
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'role': repr(self.role), f'content': repr(self.content)}
        class_name = "SessionTranscript"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'role': self.role, 'content': self.content}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
