from .audio_listener_search_term import AudioListenerSearchTerm
from .audio_listener_transcript_search_result import AudioListenerTranscriptSearchResult
from .return_class import AbstractApiClass


class AudioListenerTranscriptSearchResults(AbstractApiClass):
    """
        Search response for audio listener transcript snippets.

        Args:
            client (ApiClient): An authenticated API Client instance
            query (str): The search query that produced these results.
            total (int): The number of matching sessions.
            phrase (str): The words of the query as a phrase, when it has more than one word.
            partial (bool): Whether a statement hit its time limit and some hits are missing.
            truncated (bool): Whether the user has more sessions than the search covered.
            linesCapped (bool): Whether transcript lines were searched for the newest sessions only; titles and summaries covered them all.
            queryTerms (AudioListenerSearchTerm): The search terms the server used.
            results (AudioListenerTranscriptSearchResult): The list of matching transcript search hits.
    """

    def __init__(self, client, query=None, total=None, phrase=None, partial=None, truncated=None, linesCapped=None, queryTerms={}, results={}):
        super().__init__(client, None)
        self.query = query
        self.total = total
        self.phrase = phrase
        self.partial = partial
        self.truncated = truncated
        self.lines_capped = linesCapped
        self.query_terms = client._build_class(
            AudioListenerSearchTerm, queryTerms)
        self.results = client._build_class(
            AudioListenerTranscriptSearchResult, results)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'query': repr(self.query), f'total': repr(self.total), f'phrase': repr(self.phrase), f'partial': repr(self.partial), f'truncated': repr(
            self.truncated), f'lines_capped': repr(self.lines_capped), f'query_terms': repr(self.query_terms), f'results': repr(self.results)}
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
        resp = {'query': self.query, 'total': self.total, 'phrase': self.phrase, 'partial': self.partial, 'truncated': self.truncated,
                'lines_capped': self.lines_capped, 'query_terms': self._get_attribute_as_dict(self.query_terms), 'results': self._get_attribute_as_dict(self.results)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
