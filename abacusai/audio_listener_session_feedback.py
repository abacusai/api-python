from .return_class import AbstractApiClass


class AudioListenerSessionFeedback(AbstractApiClass):
    """
        The owner's rating of a Listener session.

        Args:
            client (ApiClient): An authenticated API Client instance
            rating (int): Stars, 1 to 5.
            comment (str): What the owner wrote with it; empty when nothing.
            atMs (int): When it was given, in ms since the epoch.
    """

    def __init__(self, client, rating=None, comment=None, atMs=None):
        super().__init__(client, None)
        self.rating = rating
        self.comment = comment
        self.at_ms = atMs
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'rating': repr(self.rating), f'comment': repr(
            self.comment), f'at_ms': repr(self.at_ms)}
        class_name = "AudioListenerSessionFeedback"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'rating': self.rating,
                'comment': self.comment, 'at_ms': self.at_ms}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
