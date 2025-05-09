from .return_class import AbstractApiClass


class CodeAutocompleteEditPredictionResponse(AbstractApiClass):
    """
        A autocomplete response from an LLM

        Args:
            client (ApiClient): An authenticated API Client instance
            autocompleteResponse (str): autocomplete code
            showAutocomplete (bool): Whether to show autocomplete in the client
    """

    def __init__(self, client, autocompleteResponse=None, showAutocomplete=None):
        super().__init__(client, None)
        self.autocomplete_response = autocompleteResponse
        self.show_autocomplete = showAutocomplete
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'autocomplete_response': repr(
            self.autocomplete_response), f'show_autocomplete': repr(self.show_autocomplete)}
        class_name = "CodeAutocompleteEditPredictionResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'autocomplete_response': self.autocomplete_response,
                'show_autocomplete': self.show_autocomplete}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
