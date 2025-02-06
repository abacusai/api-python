from .return_class import AbstractApiClass


class VoiceGenDetails(AbstractApiClass):
    """
        Voice generation details

        Args:
            client (ApiClient): An authenticated API Client instance
            model (str): The model used for voice generation.
            voice (dict): The voice details.
    """

    def __init__(self, client, model=None, voice=None):
        super().__init__(client, None)
        self.model = model
        self.voice = voice
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'model': repr(self.model), f'voice': repr(self.voice)}
        class_name = "VoiceGenDetails"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'model': self.model, 'voice': self.voice}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
