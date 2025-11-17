from .return_class import AbstractApiClass


class DomainSearchResult(AbstractApiClass):
    """
        Domain search result

        Args:
            client (ApiClient): An authenticated API Client instance
            domain (str): name of the domain in a search result
            registerCredits (int): number of credits required for registration
            renewalCredits (int): number of credits required for renewal
    """

    def __init__(self, client, domain=None, registerCredits=None, renewalCredits=None):
        super().__init__(client, None)
        self.domain = domain
        self.register_credits = registerCredits
        self.renewal_credits = renewalCredits
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'domain': repr(self.domain), f'register_credits': repr(
            self.register_credits), f'renewal_credits': repr(self.renewal_credits)}
        class_name = "DomainSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'domain': self.domain, 'register_credits': self.register_credits,
                'renewal_credits': self.renewal_credits}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
