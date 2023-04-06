from .return_class import AbstractApiClass


class ApiEndpoint(AbstractApiClass):
    """
        An collection of endpoints which can be used to make requests to, such as api calls or predict calls

        Args:
            client (ApiClient): An authenticated API Client instance
            apiEndpoint (str): The URI that can be used to make API calls
            predictEndpoint (str): The URI that can be used to make predict calls against Deployments
    """

    def __init__(self, client, apiEndpoint=None, predictEndpoint=None):
        super().__init__(client, None)
        self.api_endpoint = apiEndpoint
        self.predict_endpoint = predictEndpoint

    def __repr__(self):
        return f"ApiEndpoint(api_endpoint={repr(self.api_endpoint)},\n  predict_endpoint={repr(self.predict_endpoint)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'api_endpoint': self.api_endpoint, 'predict_endpoint': self.predict_endpoint}
