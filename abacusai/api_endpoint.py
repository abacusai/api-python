from .return_class import AbstractApiClass


class ApiEndpoint(AbstractApiClass):
    """
        An collection of endpoints which can be used to make requests to, such as api calls or predict calls

        Args:
            client (ApiClient): An authenticated API Client instance
            apiEndpoint (str): The URI that can be used to make API calls
            predictEndpoint (str): The URI that can be used to make predict calls against Deployments
            proxyEndpoint (str): The URI that can be used to make proxy server calls
    """

    def __init__(self, client, apiEndpoint=None, predictEndpoint=None, proxyEndpoint=None):
        super().__init__(client, None)
        self.api_endpoint = apiEndpoint
        self.predict_endpoint = predictEndpoint
        self.proxy_endpoint = proxyEndpoint

    def __repr__(self):
        repr_dict = {f'api_endpoint': repr(self.api_endpoint), f'predict_endpoint': repr(
            self.predict_endpoint), f'proxy_endpoint': repr(self.proxy_endpoint)}
        class_name = "ApiEndpoint"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'api_endpoint': self.api_endpoint, 'predict_endpoint':
                self.predict_endpoint, 'proxy_endpoint': self.proxy_endpoint}
        return {key: value for key, value in resp.items() if value is not None}
