from .return_class import AbstractApiClass


class AudioListenerScreen(AbstractApiClass):
    """
        A key screen of a Listener session: a frame the desktop app kept when the screen changed.

        Args:
            client (ApiClient): An authenticated API Client instance
            screenId (str): The screen's id.
            capturedAtMs (int): When it was on screen, in ms since the epoch.
            windowTitle (str): The window in front then.
            text (str): The start of the text the app read on it.
            url (str): The image, for an hour.
    """

    def __init__(self, client, screenId=None, capturedAtMs=None, windowTitle=None, text=None, url=None):
        super().__init__(client, None)
        self.screen_id = screenId
        self.captured_at_ms = capturedAtMs
        self.window_title = windowTitle
        self.text = text
        self.url = url
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'screen_id': repr(self.screen_id), f'captured_at_ms': repr(
            self.captured_at_ms), f'window_title': repr(self.window_title), f'text': repr(self.text), f'url': repr(self.url)}
        class_name = "AudioListenerScreen"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'screen_id': self.screen_id, 'captured_at_ms': self.captured_at_ms,
                'window_title': self.window_title, 'text': self.text, 'url': self.url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
