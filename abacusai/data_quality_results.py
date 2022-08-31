from .return_class import AbstractApiClass


class DataQualityResults(AbstractApiClass):
    """
        Data Quality results from normalization stage

        Args:
            client (ApiClient): An authenticated API Client instance
            results (dict): A list with different pairs of quality parameters and their values
            success (bool): 
    """

    def __init__(self, client, results=None, success=None):
        super().__init__(client, None)
        self.results = results
        self.success = success

    def __repr__(self):
        return f"DataQualityResults(results={repr(self.results)},\n  success={repr(self.success)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'results': self.results, 'success': self.success}
