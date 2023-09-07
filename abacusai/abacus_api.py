from .return_class import AbstractApiClass


class AbacusApi(AbstractApiClass):
    """
        An Abacus API.

        Args:
            client (ApiClient): An authenticated API Client instance
            method (str): The name of of the API method.
            docstring (str): The docstring of the API method.
    """

    def __init__(self, client, method=None, docstring=None):
        super().__init__(client, None)
        self.method = method
        self.docstring = docstring

    def __repr__(self):
        return f"AbacusApi(method={repr(self.method)},\n  docstring={repr(self.docstring)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'method': self.method, 'docstring': self.docstring}
