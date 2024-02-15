from .return_class import AbstractApiClass


class LlmParameters(AbstractApiClass):
    """
        The parameters of LLM for given inputs.

        Args:
            client (ApiClient): An authenticated API Client instance
            parameters (dict): The parameters of LLM for given inputs.
    """

    def __init__(self, client, parameters=None):
        super().__init__(client, None)
        self.parameters = parameters
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'parameters': repr(self.parameters)}
        class_name = "LlmParameters"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'parameters': self.parameters}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
