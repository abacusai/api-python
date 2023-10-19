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
        repr_dict = {f'method': repr(self.method),
                     f'docstring': repr(self.docstring)}
        class_name = "AbacusApi"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'method': self.method, 'docstring': self.docstring}
        return {key: value for key, value in resp.items() if value is not None}
