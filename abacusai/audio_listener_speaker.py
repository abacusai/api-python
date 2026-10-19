from .return_class import AbstractApiClass


class AudioListenerSpeaker(AbstractApiClass):
    """
        The name the user gave far-end lines of a Listener session.

        Args:
            client (ApiClient): An authenticated API Client instance
            lineIds (list[int]): The lines named.
            speaker (str): The name as it shows; none when it was taken back.
            count (int): How many stored lines changed.
            version (int): The session's edits version after the change, which a later save must send.
            editedAt (str): When the change landed.
    """

    def __init__(self, client, lineIds=None, speaker=None, count=None, version=None, editedAt=None):
        super().__init__(client, None)
        self.line_ids = lineIds
        self.speaker = speaker
        self.count = count
        self.version = version
        self.edited_at = editedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'line_ids': repr(self.line_ids), f'speaker': repr(self.speaker), f'count': repr(
            self.count), f'version': repr(self.version), f'edited_at': repr(self.edited_at)}
        class_name = "AudioListenerSpeaker"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'line_ids': self.line_ids, 'speaker': self.speaker,
                'count': self.count, 'version': self.version, 'edited_at': self.edited_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
