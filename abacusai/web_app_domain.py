from .return_class import AbstractApiClass


class WebAppDomain(AbstractApiClass):
    """
        Web App Domain

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppDomainId (id): The ID of the web app domain
            hostname (str): The hostname of the web app domain
            domainType (str): The type of the web app domain
            lifecycle (str): The lifecycle of the web app domain
            nameservers (list): The nameservers of the web app domain
    """

    def __init__(self, client, webAppDomainId=None, hostname=None, domainType=None, lifecycle=None, nameservers=None):
        super().__init__(client, webAppDomainId)
        self.web_app_domain_id = webAppDomainId
        self.hostname = hostname
        self.domain_type = domainType
        self.lifecycle = lifecycle
        self.nameservers = nameservers
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_domain_id': repr(self.web_app_domain_id), f'hostname': repr(self.hostname), f'domain_type': repr(
            self.domain_type), f'lifecycle': repr(self.lifecycle), f'nameservers': repr(self.nameservers)}
        class_name = "WebAppDomain"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_domain_id': self.web_app_domain_id, 'hostname': self.hostname,
                'domain_type': self.domain_type, 'lifecycle': self.lifecycle, 'nameservers': self.nameservers}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
