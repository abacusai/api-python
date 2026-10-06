from .return_class import AbstractApiClass


class AudioListenerCarryOver(AbstractApiClass):
    """
        What of a calendar meeting's last meeting carries over into it, in the Listener (per meeting: for a recurring one, its series).

        Args:
            client (ApiClient): An authenticated API Client instance
            enabled (bool): Whether the meeting gets what its last meeting left open.
            instructions (str): The user's own words for what carries over; empty for the default.
            defaultInstructions (str): What carries over without instructions.
            maxChars (int): The longest instructions allowed.
    """

    def __init__(self, client, enabled=None, instructions=None, defaultInstructions=None, maxChars=None):
        super().__init__(client, None)
        self.enabled = enabled
        self.instructions = instructions
        self.default_instructions = defaultInstructions
        self.max_chars = maxChars
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'enabled': repr(self.enabled), f'instructions': repr(
            self.instructions), f'default_instructions': repr(self.default_instructions), f'max_chars': repr(self.max_chars)}
        class_name = "AudioListenerCarryOver"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'enabled': self.enabled, 'instructions': self.instructions,
                'default_instructions': self.default_instructions, 'max_chars': self.max_chars}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
