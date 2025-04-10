from .return_class import AbstractApiClass


class CodeSummaryResponse(AbstractApiClass):
    """
        A summary response from an LLM

        Args:
            client (ApiClient): An authenticated API Client instance
            summary (str): The summary of the code.
    """

    def __init__(self, client, summary=None):
        super().__init__(client, None)
        self.summary = summary
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'summary': repr(self.summary)}
        class_name = "CodeSummaryResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'summary': self.summary}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
