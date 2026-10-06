from .return_class import AbstractApiClass


class AudioListenerMeetingSessions(AbstractApiClass):
    """
        The user's recent calendar meetings in the Listener, each with its sessions.

        Args:
            client (ApiClient): An authenticated API Client instance
            meetings (list): Newest first: title, provider, recurring, session_count, last_created_at_ms and the newest sessions (id, name, created_at_ms).
    """

    def __init__(self, client, meetings=None):
        super().__init__(client, None)
        self.meetings = meetings
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'meetings': repr(self.meetings)}
        class_name = "AudioListenerMeetingSessions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'meetings': self.meetings}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
