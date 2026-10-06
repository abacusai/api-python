from .return_class import AbstractApiClass


class AudioListenerMeetingBrief(AbstractApiClass):
    """
        What the user's previous Listener session of a meeting left for the next one.

        Args:
            client (ApiClient): An authenticated API Client instance
            previous (dict): The previous session of the same meeting (id, name, created_at_ms), if any.
            followUps (list): What it left open: action items, commitments and open questions.
            seriesTemplateId (str): The template the previous session ran with, when the user still has it.
            thisOccurrence (dict): The user's session of this very occurrence of the meeting (id, name, created_at_ms), if they already have one.
    """

    def __init__(self, client, previous=None, followUps=None, seriesTemplateId=None, thisOccurrence=None):
        super().__init__(client, None)
        self.previous = previous
        self.follow_ups = followUps
        self.series_template_id = seriesTemplateId
        self.this_occurrence = thisOccurrence
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'previous': repr(self.previous), f'follow_ups': repr(self.follow_ups), f'series_template_id': repr(
            self.series_template_id), f'this_occurrence': repr(self.this_occurrence)}
        class_name = "AudioListenerMeetingBrief"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'previous': self.previous, 'follow_ups': self.follow_ups,
                'series_template_id': self.series_template_id, 'this_occurrence': self.this_occurrence}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
