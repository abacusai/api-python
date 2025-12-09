from .return_class import AbstractApiClass


class DomainSearchResult(AbstractApiClass):
    """
        Domain search result

        Args:
            client (ApiClient): An authenticated API Client instance
            domain (str): name of the domain in a search result
            registerCredits (int): number of credits required for registration
            registerYears (int): number of years the domain will get registered for
            renewalCredits (int): number of credits required for renewal
            renewalYears (int): number of years the domain will be renewed for
    """

    def __init__(self, client, domain=None, registerCredits=None, registerYears=None, renewalCredits=None, renewalYears=None):
        super().__init__(client, None)
        self.domain = domain
        self.register_credits = registerCredits
        self.register_years = registerYears
        self.renewal_credits = renewalCredits
        self.renewal_years = renewalYears
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'domain': repr(self.domain), f'register_credits': repr(self.register_credits), f'register_years': repr(
            self.register_years), f'renewal_credits': repr(self.renewal_credits), f'renewal_years': repr(self.renewal_years)}
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
        resp = {'domain': self.domain, 'register_credits': self.register_credits, 'register_years': self.register_years,
                'renewal_credits': self.renewal_credits, 'renewal_years': self.renewal_years}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
