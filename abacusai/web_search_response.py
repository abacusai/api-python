from .return_class import AbstractApiClass
from .web_search_result import WebSearchResult


class WebSearchResponse(AbstractApiClass):
    """
        Result of running a web search with optional content fetching.

        Args:
            client (ApiClient): An authenticated API Client instance
            searchResults (WebSearchResult): List of search results.
    """

    def __init__(self, client, searchResults={}):
        super().__init__(client, None)
        self.search_results = client._build_class(
            WebSearchResult, searchResults)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'search_results': repr(self.search_results)}
        class_name = "WebSearchResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'search_results': self._get_attribute_as_dict(
            self.search_results)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
