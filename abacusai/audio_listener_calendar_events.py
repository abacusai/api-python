from .return_class import AbstractApiClass


class AudioListenerCalendarEvents(AbstractApiClass):
    """
        The meetings on the user's connected calendars in a window, for the Listener.

        Args:
            client (ApiClient): An authenticated API Client instance
            providers (list): Each calendar provider (google, outlook): whether it is connected, its account, and an error if it could not be read.
            events (list): The meetings in the window, by start time, from every connected calendar.
            startMs (int): The start of the window read, epoch milliseconds.
            endMs (int): The end of the window read, epoch milliseconds.
    """

    def __init__(self, client, providers=None, events=None, startMs=None, endMs=None):
        super().__init__(client, None)
        self.providers = providers
        self.events = events
        self.start_ms = startMs
        self.end_ms = endMs
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'providers': repr(self.providers), f'events': repr(
            self.events), f'start_ms': repr(self.start_ms), f'end_ms': repr(self.end_ms)}
        class_name = "AudioListenerCalendarEvents"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'providers': self.providers, 'events': self.events,
                'start_ms': self.start_ms, 'end_ms': self.end_ms}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
