from .return_class import AbstractApiClass


class AudioListenerSearchTerm(AbstractApiClass):
    """
        One search term as the server read it from the query.

        Args:
            client (ApiClient): An authenticated API Client instance
            text (str): The term as typed.
            short (bool): Whether the term matches whole words only (short ASCII terms such as AI or API) rather than any substring.
    """

    def __init__(self, client, text=None, short=None):
        super().__init__(client, None)
        self.text = text
        self.short = short
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'text': repr(self.text), f'short': repr(self.short)}
        class_name = "AudioListenerSearchTerm"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'text': self.text, 'short': self.short}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
