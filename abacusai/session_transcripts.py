from .return_class import AbstractApiClass


class SessionTranscripts(AbstractApiClass):
    """
        A list of session transcripts

        Args:
            client (ApiClient): An authenticated API Client instance
            transcripts (list[sessiontranscript]): A list of session transcripts.
    """

    def __init__(self, client, transcripts=None):
        super().__init__(client, None)
        self.transcripts = transcripts
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'transcripts': repr(self.transcripts)}
        class_name = "SessionTranscripts"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'transcripts': self.transcripts}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
