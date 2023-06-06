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

    def __repr__(self):
        return f"LlmParameters(parameters={repr(self.parameters)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'parameters': self.parameters}
