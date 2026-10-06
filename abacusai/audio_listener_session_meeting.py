from .audio_listener_meeting_brief import AudioListenerMeetingBrief
from .return_class import AbstractApiClass


class AudioListenerSessionMeeting(AbstractApiClass):
    """
        The calendar event a Listener session is attached to, and the brief of its previous meeting.

        Args:
            client (ApiClient): An authenticated API Client instance
            meeting (dict): The event as the session keeps it (event_ref, provider, title, start_ms, end_ms, recurring); null when detached.
            brief (AudioListenerMeetingBrief): The previous session of the same meeting and what it left open.
    """

    def __init__(self, client, meeting=None, brief={}):
        super().__init__(client, None)
        self.meeting = meeting
        self.brief = client._build_class(AudioListenerMeetingBrief, brief)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'meeting': repr(
            self.meeting), f'brief': repr(self.brief)}
        class_name = "AudioListenerSessionMeeting"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'meeting': self.meeting,
                'brief': self._get_attribute_as_dict(self.brief)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
