from .return_class import AbstractApiClass


class SlackConnectorResponse(AbstractApiClass):
    """
        The response to a slack command formatted to be readable by Slack

        Args:
            client (ApiClient): An authenticated API Client instance
            text (str): The text body of the response, which becomes the returned message
            blocks (list): List of blocks in the response
    """

    def __init__(self, client, text=None, blocks=None):
        super().__init__(client, None)
        self.text = text
        self.blocks = blocks

    def __repr__(self):
        return f"SlackConnectorResponse(text={repr(self.text)},\n  blocks={repr(self.blocks)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'text': self.text, 'blocks': self.blocks}
