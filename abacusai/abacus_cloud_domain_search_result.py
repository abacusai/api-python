from .return_class import AbstractApiClass


class AbacusCloudDomainSearchResult(AbstractApiClass):
    """
        A name the Abacus AI Cloud free domain search found available

        Args:
            client (ApiClient): An authenticated API Client instance
            domain (str): The domain name
            included (bool): Whether the free domain covers it (standard-priced names only)
    """

    def __init__(self, client, domain=None, included=None):
        super().__init__(client, None)
        self.domain = domain
        self.included = included
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'domain': repr(self.domain),
                     f'included': repr(self.included)}
        class_name = "AbacusCloudDomainSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'domain': self.domain, 'included': self.included}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
