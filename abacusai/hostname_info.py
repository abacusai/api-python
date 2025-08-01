from .return_class import AbstractApiClass


class HostnameInfo(AbstractApiClass):
    """
        Hostname Info

        Args:
            client (ApiClient): An authenticated API Client instance
            isRootDomain (bool): Whether the hostname is a root domain
    """

    def __init__(self, client, isRootDomain=None):
        super().__init__(client, None)
        self.is_root_domain = isRootDomain
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'is_root_domain': repr(self.is_root_domain)}
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
        resp = {'is_root_domain': self.is_root_domain}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
