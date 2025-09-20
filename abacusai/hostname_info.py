from .return_class import AbstractApiClass


class HostnameInfo(AbstractApiClass):
    """
        Hostname Info

        Args:
            client (ApiClient): An authenticated API Client instance
            isRootDomain (bool): Whether the hostname is a root domain
            registrar (str): The registrar of the domain
            suggestedFlow (str): Suggested flow for the domain
            isAutomaticFlowAvailable (bool): Whether entri is supported for the domain
    """

    def __init__(self, client, isRootDomain=None, registrar=None, suggestedFlow=None, isAutomaticFlowAvailable=None):
        super().__init__(client, None)
        self.is_root_domain = isRootDomain
        self.registrar = registrar
        self.suggested_flow = suggestedFlow
        self.is_automatic_flow_available = isAutomaticFlowAvailable
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'is_root_domain': repr(self.is_root_domain), f'registrar': repr(self.registrar), f'suggested_flow': repr(
            self.suggested_flow), f'is_automatic_flow_available': repr(self.is_automatic_flow_available)}
        class_name = "HostnameInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'is_root_domain': self.is_root_domain, 'registrar': self.registrar,
                'suggested_flow': self.suggested_flow, 'is_automatic_flow_available': self.is_automatic_flow_available}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
