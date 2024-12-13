from .return_class import AbstractApiClass


class WebPageResponse(AbstractApiClass):
    """
        A scraped web page response

        Args:
            client (ApiClient): An authenticated API Client instance
            content (str): The content of the web page.
    """

    def __init__(self, client, content=None):
        super().__init__(client, None)
        self.content = content
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'content': repr(self.content)}
        class_name = "WebPageResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'content': self.content}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
