from .return_class import AbstractApiClass


class LlmCodeBlock(AbstractApiClass):
    """
        Parsed code block from an LLM response

        Args:
            client (ApiClient): An authenticated API Client instance
            language (str): The language of the code block. Eg - python/sql/etc.
            code (str): source code string
            start (int): index of the starting character of the code block in the original response
            end (int): index of the last character of the code block in the original response
            valid (bool): flag denoting whether the soruce code string is syntactically valid
    """

    def __init__(self, client, language=None, code=None, start=None, end=None, valid=None):
        super().__init__(client, None)
        self.language = language
        self.code = code
        self.start = start
        self.end = end
        self.valid = valid
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'language': repr(self.language), f'code': repr(self.code), f'start': repr(
            self.start), f'end': repr(self.end), f'valid': repr(self.valid)}
        class_name = "LlmCodeBlock"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'language': self.language, 'code': self.code,
                'start': self.start, 'end': self.end, 'valid': self.valid}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
