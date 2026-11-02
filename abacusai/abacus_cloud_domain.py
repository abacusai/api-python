from .return_class import AbstractApiClass
from .web_app_domain import WebAppDomain


class AbacusCloudDomain(AbstractApiClass):
    """
        The Abacus AI Cloud subscriber's free custom domain

        Args:
            client (ApiClient): An authenticated API Client instance
            paid (bool): Whether the subscription has billed (a claim while trialing ends the trial and charges the first month that day)
            claimable (bool): Not yet claimed
            maxRegisterCredits (int): The highest registration price (credits) the free domain covers
            domain (WebAppDomain): The claimed domain (absent until claimed)
    """

    def __init__(self, client, paid=None, claimable=None, maxRegisterCredits=None, domain={}):
        super().__init__(client, None)
        self.paid = paid
        self.claimable = claimable
        self.max_register_credits = maxRegisterCredits
        self.domain = client._build_class(WebAppDomain, domain)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'paid': repr(self.paid), f'claimable': repr(self.claimable), f'max_register_credits': repr(
            self.max_register_credits), f'domain': repr(self.domain)}
        class_name = "AbacusCloudDomain"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'paid': self.paid, 'claimable': self.claimable, 'max_register_credits':
                self.max_register_credits, 'domain': self._get_attribute_as_dict(self.domain)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
