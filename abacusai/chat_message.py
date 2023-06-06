from .return_class import AbstractApiClass


class ChatMessage(AbstractApiClass):
    """
        A single chat message with Abacus Chat.

        Args:
            client (ApiClient): An authenticated API Client instance
            role (str): The role of the message sender
            text (list): A list of text segments for the message
            timestamp (str): The timestamp at which the message was sent
            isUseful (bool): Whether this message was marked as useful or not
            feedback (str): The feedback provided for the message
    """

    def __init__(self, client, role=None, text=None, timestamp=None, isUseful=None, feedback=None):
        super().__init__(client, None)
        self.role = role
        self.text = text
        self.timestamp = timestamp
        self.is_useful = isUseful
        self.feedback = feedback

    def __repr__(self):
        return f"ChatMessage(role={repr(self.role)},\n  text={repr(self.text)},\n  timestamp={repr(self.timestamp)},\n  is_useful={repr(self.is_useful)},\n  feedback={repr(self.feedback)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'role': self.role, 'text': self.text, 'timestamp': self.timestamp, 'is_useful': self.is_useful, 'feedback': self.feedback}
