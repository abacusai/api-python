from .return_class import AbstractApiClass


class LlmInput(AbstractApiClass):
    """
        The result of encoding an object as input for a language model.

        Args:
            client (ApiClient): An authenticated API Client instance
            content (str): Content of the response
    """

    def __init__(self, client, content=None):
        super().__init__(client, None)
        self.content = content

    def __repr__(self):
        return f"LlmInput(content={repr(self.content)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'content': self.content}
