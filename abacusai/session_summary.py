from .return_class import AbstractApiClass


class SessionSummary(AbstractApiClass):
    """
        A session summary

        Args:
            client (ApiClient): An authenticated API Client instance
            summary (str): The summary of the session.
            templateName (str): The name of the Listener template the session ran with, if any.
            meetingType (str): The meeting type of that template.
            meetingNotes (str): The session's notes for this meeting, returned only to the session's owner.
            meeting (dict): The calendar event the session is attached to (title, times, provider, recurring), returned only to the owner.
            previousMeeting (dict): The owner's previous session of the same meeting (id, name, created_at_ms), returned only to the owner.
            feedback (dict): The owner's rating of the session (rating 1 to 5, comment, at_ms), returned only to the owner.
    """

    def __init__(self, client, summary=None, templateName=None, meetingType=None, meetingNotes=None, meeting=None, previousMeeting=None, feedback=None):
        super().__init__(client, None)
        self.summary = summary
        self.template_name = templateName
        self.meeting_type = meetingType
        self.meeting_notes = meetingNotes
        self.meeting = meeting
        self.previous_meeting = previousMeeting
        self.feedback = feedback
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'summary': repr(self.summary), f'template_name': repr(self.template_name), f'meeting_type': repr(self.meeting_type), f'meeting_notes': repr(
            self.meeting_notes), f'meeting': repr(self.meeting), f'previous_meeting': repr(self.previous_meeting), f'feedback': repr(self.feedback)}
        class_name = "SessionSummary"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'summary': self.summary, 'template_name': self.template_name, 'meeting_type': self.meeting_type,
                'meeting_notes': self.meeting_notes, 'meeting': self.meeting, 'previous_meeting': self.previous_meeting, 'feedback': self.feedback}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
