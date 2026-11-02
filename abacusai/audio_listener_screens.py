from .audio_listener_screen import AudioListenerScreen
from .return_class import AbstractApiClass


class AudioListenerScreens(AbstractApiClass):
    """
        The key screens of a Listener session, oldest first.

        Args:
            client (ApiClient): An authenticated API Client instance
            limit (int): The most a session keeps.
            screens (AudioListenerScreen): The screens.
    """

    def __init__(self, client, limit=None, screens={}):
        super().__init__(client, None)
        self.limit = limit
        self.screens = client._build_class(AudioListenerScreen, screens)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'limit': repr(self.limit),
                     f'screens': repr(self.screens)}
        class_name = "AudioListenerScreens"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'limit': self.limit,
                'screens': self._get_attribute_as_dict(self.screens)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
