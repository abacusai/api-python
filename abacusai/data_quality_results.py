from .return_class import AbstractApiClass


class DataQualityResults(AbstractApiClass):
    """
        Data Quality results from normalization stage

        Args:
            client (ApiClient): An authenticated API Client instance
            results (dict): A list with different pairs of quality parameters and their values
    """

    def __init__(self, client, results=None):
        super().__init__(client, None)
        self.results = results
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'results': repr(self.results)}
        class_name = "DataQualityResults"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'results': self.results}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
