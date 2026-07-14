from .audio_listener_transcript_search_result import AudioListenerTranscriptSearchResult
from .return_class import AbstractApiClass


class AudioListenerTranscriptSearchResults(AbstractApiClass):
    """
        Search response for audio listener transcript snippets.

        Args:
            client (ApiClient): An authenticated API Client instance
            query (str): The search query that produced these results.
            total (int): The total number of matching results.
            results (AudioListenerTranscriptSearchResult): The list of matching transcript search hits.
    """

    def __init__(self, client, query=None, total=None, results={}):
        super().__init__(client, None)
        self.query = query
        self.total = total
        self.results = client._build_class(
            AudioListenerTranscriptSearchResult, results)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'query': repr(self.query), f'total': repr(
            self.total), f'results': repr(self.results)}
        class_name = "AudioListenerTranscriptSearchResults"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'query': self.query, 'total': self.total,
                'results': self._get_attribute_as_dict(self.results)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
