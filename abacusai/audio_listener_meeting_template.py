from .return_class import AbstractApiClass


class AudioListenerMeetingTemplate(AbstractApiClass):
    """
        The template a calendar meeting starts with in the Listener.

        Args:
            client (ApiClient): An authenticated API Client instance
            templateId (str): The template the user chose for the meeting (for a recurring one, its series); absent when none is chosen.
    """

    def __init__(self, client, templateId=None):
        super().__init__(client, None)
        self.template_id = templateId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'template_id': repr(self.template_id)}
        class_name = "AudioListenerMeetingTemplate"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'template_id': self.template_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
