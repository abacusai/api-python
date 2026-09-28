from .audio_listener_template_file import AudioListenerTemplateFile
from .return_class import AbstractApiClass


class AudioListenerTemplate(AbstractApiClass):
    """
        A Listener session template: a built-in, possibly edited by the user, or one of the user's own.

        Args:
            client (ApiClient): An authenticated API Client instance
            templateId (str): builtin:<key> for a built-in, else the template's id.
            builtinKey (str): The built-in's key; absent for the user's own templates.
            name (str): The template's name.
            meetingType (str): general, meetings, customers, hiring, learning or custom.
            meetingBrief (str): What every job should know about this kind of meeting.
            insightGuidelines (str): What insights focus on.
            insightIntervalSecs (int): Seconds between insights; 0 turns them off.
            summaryFormat (str): How the summary is written; empty for the default structure.
            isHidden (bool): A built-in the user hid.
            isModified (bool): A built-in the user changed.
            createdAt (str): When the user's row was created; absent for an unchanged built-in.
            updatedAt (str): When the user's row last changed.
            files (AudioListenerTemplateFile): The template's reference files, oldest first.
    """

    def __init__(self, client, templateId=None, builtinKey=None, name=None, meetingType=None, meetingBrief=None, insightGuidelines=None, insightIntervalSecs=None, summaryFormat=None, isHidden=None, isModified=None, createdAt=None, updatedAt=None, files={}):
        super().__init__(client, None)
        self.template_id = templateId
        self.builtin_key = builtinKey
        self.name = name
        self.meeting_type = meetingType
        self.meeting_brief = meetingBrief
        self.insight_guidelines = insightGuidelines
        self.insight_interval_secs = insightIntervalSecs
        self.summary_format = summaryFormat
        self.is_hidden = isHidden
        self.is_modified = isModified
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.files = client._build_class(AudioListenerTemplateFile, files)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'template_id': repr(self.template_id), f'builtin_key': repr(self.builtin_key), f'name': repr(self.name), f'meeting_type': repr(self.meeting_type), f'meeting_brief': repr(self.meeting_brief), f'insight_guidelines': repr(self.insight_guidelines), f'insight_interval_secs': repr(
            self.insight_interval_secs), f'summary_format': repr(self.summary_format), f'is_hidden': repr(self.is_hidden), f'is_modified': repr(self.is_modified), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'files': repr(self.files)}
        class_name = "AudioListenerTemplate"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'template_id': self.template_id, 'builtin_key': self.builtin_key, 'name': self.name, 'meeting_type': self.meeting_type, 'meeting_brief': self.meeting_brief, 'insight_guidelines': self.insight_guidelines, 'insight_interval_secs':
                self.insight_interval_secs, 'summary_format': self.summary_format, 'is_hidden': self.is_hidden, 'is_modified': self.is_modified, 'created_at': self.created_at, 'updated_at': self.updated_at, 'files': self._get_attribute_as_dict(self.files)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
