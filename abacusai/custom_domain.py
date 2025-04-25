from .return_class import AbstractApiClass


class CustomDomain(AbstractApiClass):
    """
        Result of adding a custom domain to a hosted app

        Args:
            client (ApiClient): An authenticated API Client instance
            status (bool): Whether the custom domain was added successfully
            message (str): The message from the custom domain
            expectedNameservers (list): The expected nameservers for the custom domain
            currentNameservers (list): The current nameservers for the custom domain
    """

    def __init__(self, client, status=None, message=None, expectedNameservers=None, currentNameservers=None):
        super().__init__(client, None)
        self.status = status
        self.message = message
        self.expected_nameservers = expectedNameservers
        self.current_nameservers = currentNameservers
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'status': repr(self.status), f'message': repr(self.message), f'expected_nameservers': repr(
            self.expected_nameservers), f'current_nameservers': repr(self.current_nameservers)}
        class_name = "CustomDomain"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'status': self.status, 'message': self.message, 'expected_nameservers':
                self.expected_nameservers, 'current_nameservers': self.current_nameservers}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
