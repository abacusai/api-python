from .return_class import AbstractApiClass


class DomainConnectCheck(AbstractApiClass):
    """
        Whether the DNS provider has applied the Domain Connect records

        Args:
            client (ApiClient): An authenticated API Client instance
            configured (bool): The domain's nameservers serve our records, and the domain is marked as configured
            checked (bool): Whether the nameservers answered; false when the lookup did not complete, so configured says nothing yet
    """

    def __init__(self, client, configured=None, checked=None):
        super().__init__(client, None)
        self.configured = configured
        self.checked = checked
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'configured': repr(
            self.configured), f'checked': repr(self.checked)}
        class_name = "DomainConnectCheck"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'configured': self.configured, 'checked': self.checked}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
