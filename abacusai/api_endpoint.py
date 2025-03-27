from .return_class import AbstractApiClass


class ApiEndpoint(AbstractApiClass):
    """
        An collection of endpoints which can be used to make requests to, such as api calls or predict calls

        Args:
            client (ApiClient): An authenticated API Client instance
            apiEndpoint (str): The URI that can be used to make API calls
            predictEndpoint (str): The URI that can be used to make predict calls against Deployments
            proxyEndpoint (str): The URI that can be used to make proxy server calls
            llmEndpoint (str): The URI that can be used to make llm api calls
            externalChatEndpoint (str): The URI that can be used to access the external chat
            dashboardEndpoint (str):  The URI that the external chat will use to go back to the dashboard
            hostingDomain (str): The domain for hosted app deployments
    """

    def __init__(self, client, apiEndpoint=None, predictEndpoint=None, proxyEndpoint=None, llmEndpoint=None, externalChatEndpoint=None, dashboardEndpoint=None, hostingDomain=None):
        super().__init__(client, None)
        self.api_endpoint = apiEndpoint
        self.predict_endpoint = predictEndpoint
        self.proxy_endpoint = proxyEndpoint
        self.llm_endpoint = llmEndpoint
        self.external_chat_endpoint = externalChatEndpoint
        self.dashboard_endpoint = dashboardEndpoint
        self.hosting_domain = hostingDomain
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'api_endpoint': repr(self.api_endpoint), f'predict_endpoint': repr(self.predict_endpoint), f'proxy_endpoint': repr(self.proxy_endpoint), f'llm_endpoint': repr(
            self.llm_endpoint), f'external_chat_endpoint': repr(self.external_chat_endpoint), f'dashboard_endpoint': repr(self.dashboard_endpoint), f'hosting_domain': repr(self.hosting_domain)}
        class_name = "ApiEndpoint"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'api_endpoint': self.api_endpoint, 'predict_endpoint': self.predict_endpoint, 'proxy_endpoint': self.proxy_endpoint, 'llm_endpoint': self.llm_endpoint,
                'external_chat_endpoint': self.external_chat_endpoint, 'dashboard_endpoint': self.dashboard_endpoint, 'hosting_domain': self.hosting_domain}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
