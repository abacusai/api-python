from .return_class import AbstractApiClass


class AudioListenerTemplateSelection(AbstractApiClass):
    """
        The template a user's next Listener session starts with, and the ones they used recently.

        Args:
            client (ApiClient): An authenticated API Client instance
            selectedTemplateId (str): The selected template.
            recentTemplateIds (list): Recently used templates, newest first.
    """

    def __init__(self, client, selectedTemplateId=None, recentTemplateIds=None):
        super().__init__(client, None)
        self.selected_template_id = selectedTemplateId
        self.recent_template_ids = recentTemplateIds
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'selected_template_id': repr(
            self.selected_template_id), f'recent_template_ids': repr(self.recent_template_ids)}
        class_name = "AudioListenerTemplateSelection"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'selected_template_id': self.selected_template_id,
                'recent_template_ids': self.recent_template_ids}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
