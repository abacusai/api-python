from .return_class import AbstractApiClass


class ChatMessage(AbstractApiClass):
    """
        A single chat message with Abacus Chat.

        Args:
            client (ApiClient): An authenticated API Client instance
            role (str): The role of the message sender
            text (list): A list of text segments for the message
            timestamp (str): The timestamp at which the message was sent
    """

    def __init__(self, client, role=None, text=None, timestamp=None):
        super().__init__(client, None)
        self.role = role
        self.text = text
        self.timestamp = timestamp

    def __repr__(self):
        return f"ChatMessage(role={repr(self.role)},\n  text={repr(self.text)},\n  timestamp={repr(self.timestamp)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'role': self.role, 'text': self.text, 'timestamp': self.timestamp}
