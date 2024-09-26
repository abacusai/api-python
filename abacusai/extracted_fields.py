from .return_class import AbstractApiClass


class ExtractedFields(AbstractApiClass):
    """
        The fields extracted from a document.

        Args:
            client (ApiClient): An authenticated API Client instance
            data (dict): The fields/data extracted from the document.
            rawLlmResponse (str): The raw llm response. Only returned if it could not be parsed to json dict.
    """

    def __init__(self, client, data=None, rawLlmResponse=None):
        super().__init__(client, None)
        self.data = data
        self.raw_llm_response = rawLlmResponse
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'data': repr(
            self.data), f'raw_llm_response': repr(self.raw_llm_response)}
        class_name = "ExtractedFields"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'data': self.data, 'raw_llm_response': self.raw_llm_response}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
