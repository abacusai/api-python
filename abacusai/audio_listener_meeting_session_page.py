from .return_class import AbstractApiClass


class AudioListenerMeetingSessionPage(AbstractApiClass):
    """
        A page of every session of one calendar meeting in the Listener.

        Args:
            client (ApiClient): An authenticated API Client instance
            meeting (dict): The meeting (title, provider, recurring, session_count, last_created_at_ms) with this page of its sessions, newest first; null when the session belongs to no recent meeting.
            start (int): The index of the page's first session.
    """

    def __init__(self, client, meeting=None, start=None):
        super().__init__(client, None)
        self.meeting = meeting
        self.start = start
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'meeting': repr(
            self.meeting), f'start': repr(self.start)}
        class_name = "AudioListenerMeetingSessionPage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'meeting': self.meeting, 'start': self.start}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
