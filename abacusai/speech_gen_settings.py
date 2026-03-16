from .return_class import AbstractApiClass


class SpeechGenSettings(AbstractApiClass):
    """
        Unified speech generation settings combining STT, TTS, and STS.

        Args:
            client (ApiClient): An authenticated API Client instance
            speechType (dict): Dropdown for type of speech (speech_to_text, text_to_speech, speech_to_speech).
            modelsByType (dict): Maps each speech type to the list of applicable model keys.
            settingsByType (dict): Maps each speech type to its model settings dict.
    """

    def __init__(self, client, speechType=None, modelsByType=None, settingsByType=None):
        super().__init__(client, None)
        self.speech_type = speechType
        self.models_by_type = modelsByType
        self.settings_by_type = settingsByType
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'speech_type': repr(self.speech_type), f'models_by_type': repr(
            self.models_by_type), f'settings_by_type': repr(self.settings_by_type)}
        class_name = "SpeechGenSettings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'speech_type': self.speech_type, 'models_by_type': self.models_by_type,
                'settings_by_type': self.settings_by_type}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
