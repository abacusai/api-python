from .return_class import AbstractApiClass


class WebSearchResult(AbstractApiClass):
    """
        A single search result.

        Args:
            client (ApiClient): An authenticated API Client instance
            title (str): The title of the search result.
            url (str): The URL of the search result.
            snippet (str): The snippet of the search result.
            content (str): The page of content fetched from the url.
    """

    def __init__(self, client, title=None, url=None, snippet=None, content=None):
        super().__init__(client, None)
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'title': repr(self.title), f'url': repr(
            self.url), f'snippet': repr(self.snippet), f'content': repr(self.content)}
        class_name = "WebSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'title': self.title, 'url': self.url,
                'snippet': self.snippet, 'content': self.content}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
