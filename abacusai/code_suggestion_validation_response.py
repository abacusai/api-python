from .return_class import AbstractApiClass


class CodeSuggestionValidationResponse(AbstractApiClass):
    """
        A response from an LLM to validate a code suggestion.

        Args:
            client (ApiClient): An authenticated API Client instance
            isValid (bool): Whether the code suggestion is valid.
    """

    def __init__(self, client, isValid=None):
        super().__init__(client, None)
        self.is_valid = isValid
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'is_valid': repr(self.is_valid)}
        class_name = "CodeSuggestionValidationResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'is_valid': self.is_valid}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
