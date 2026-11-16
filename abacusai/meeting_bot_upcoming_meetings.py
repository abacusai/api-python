from .return_class import AbstractApiClass


class MeetingBotUpcomingMeetings(AbstractApiClass):
    """
        The next week's calendar meetings the notetaker can join.

        Args:
            client (ApiClient): An authenticated API Client instance
            providers (list): Each calendar provider (google, outlook): whether it is connected, and an error if it could not be read.
            events (list): The meetings with a supported link, by start time.
            botEmail (str): The notetaker's Google account, the address to invite to a meeting; absent when it joins as a guest.
    """

    def __init__(self, client, providers=None, events=None, botEmail=None):
        super().__init__(client, None)
        self.providers = providers
        self.events = events
        self.bot_email = botEmail
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'providers': repr(self.providers), f'events': repr(
            self.events), f'bot_email': repr(self.bot_email)}
        class_name = "MeetingBotUpcomingMeetings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'providers': self.providers,
                'events': self.events, 'bot_email': self.bot_email}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
