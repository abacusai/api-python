from .return_class import AbstractApiClass


class SlackConnectorResponse(AbstractApiClass):
    """
        The response to a slack command formatted to be readable by Slack

        Args:
            client (ApiClient): An authenticated API Client instance
            text (str): The text body of the response, which becomes the returned message
            attachments (list): The attachments to the text body
    """

    def __init__(self, client, text=None, attachments=None):
        super().__init__(client, None)
        self.text = text
        self.attachments = attachments

    def __repr__(self):
        return f"SlackConnectorResponse(text={repr(self.text)},\n  attachments={repr(self.attachments)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'text': self.text, 'attachments': self.attachments}
