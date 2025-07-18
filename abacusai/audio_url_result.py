from .return_class import AbstractApiClass


class AudioUrlResult(AbstractApiClass):
    """
        TTS result

        Args:
            client (ApiClient): An authenticated API Client instance
            audioUrl (str): The audio url.
            creditsUsed (float): The credits used.
    """

    def __init__(self, client, audioUrl=None, creditsUsed=None):
        super().__init__(client, None)
        self.audio_url = audioUrl
        self.credits_used = creditsUsed
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'audio_url': repr(
            self.audio_url), f'credits_used': repr(self.credits_used)}
        class_name = "AudioUrlResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'audio_url': self.audio_url, 'credits_used': self.credits_used}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
