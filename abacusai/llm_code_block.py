from .return_class import AbstractApiClass


class LlmCodeBlock(AbstractApiClass):
    """
        Parsed code block from an LLM response

        Args:
            client (ApiClient): An authenticated API Client instance
            language (str): The language of the code block. Eg - python/SQL/etc.
            code (str): source code string
            start (number(integer)): index of the starting character of the code block in the original response
            end (number(integer)): index of the last character of the code block in the original response
            valid (bool): flag denoting whether the soruce code string is syntactically valid
    """

    def __init__(self, client, language=None, code=None, start=None, end=None, valid=None):
        super().__init__(client, None)
        self.language = language
        self.code = code
        self.start = start
        self.end = end
        self.valid = valid

    def __repr__(self):
        return f"LlmCodeBlock(language={repr(self.language)},\n  code={repr(self.code)},\n  start={repr(self.start)},\n  end={repr(self.end)},\n  valid={repr(self.valid)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'language': self.language, 'code': self.code, 'start': self.start, 'end': self.end, 'valid': self.valid}
