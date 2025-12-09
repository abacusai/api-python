from .return_class import AbstractApiClass


class HostnameInfo(AbstractApiClass):
    """
        Hostname Info

        Args:
            client (ApiClient): An authenticated API Client instance
            isRootDomain (bool): Whether the hostname is a root domain
            hasRootNameserversConfigured (bool): Whether the root domain has Abacus nameservers configured.
    """

    def __init__(self, client, isRootDomain=None, hasRootNameserversConfigured=None):
        super().__init__(client, None)
        self.is_root_domain = isRootDomain
        self.has_root_nameservers_configured = hasRootNameserversConfigured
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'is_root_domain': repr(self.is_root_domain), f'has_root_nameservers_configured': repr(
            self.has_root_nameservers_configured)}
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
        resp = {'is_root_domain': self.is_root_domain,
                'has_root_nameservers_configured': self.has_root_nameservers_configured}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
