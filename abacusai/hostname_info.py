from .return_class import AbstractApiClass


class HostnameInfo(AbstractApiClass):
    """
        Hostname Info

        Args:
            client (ApiClient): An authenticated API Client instance
            hostname (str): IDN formatted hostname
            isRootDomain (bool): Whether the hostname is a root domain
            hasRootNameserversConfigured (bool): Whether the root domain has Abacus nameservers configured.
            cloudflareOauthAvailable (bool): Whether the Cloudflare OAuth flow is offered: the domain is hosted on Cloudflare and Domain Connect is not offered for it
            domainConnectProvider (str): Display name of the DNS provider when the domain can be configured through Domain Connect, else null. At most one of the two paths is offered
    """

    def __init__(self, client, hostname=None, isRootDomain=None, hasRootNameserversConfigured=None, cloudflareOauthAvailable=None, domainConnectProvider=None):
        super().__init__(client, None)
        self.hostname = hostname
        self.is_root_domain = isRootDomain
        self.has_root_nameservers_configured = hasRootNameserversConfigured
        self.cloudflare_oauth_available = cloudflareOauthAvailable
        self.domain_connect_provider = domainConnectProvider
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hostname': repr(self.hostname), f'is_root_domain': repr(self.is_root_domain), f'has_root_nameservers_configured': repr(
            self.has_root_nameservers_configured), f'cloudflare_oauth_available': repr(self.cloudflare_oauth_available), f'domain_connect_provider': repr(self.domain_connect_provider)}
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
        resp = {'hostname': self.hostname, 'is_root_domain': self.is_root_domain, 'has_root_nameservers_configured': self.has_root_nameservers_configured,
                'cloudflare_oauth_available': self.cloudflare_oauth_available, 'domain_connect_provider': self.domain_connect_provider}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
