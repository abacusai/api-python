from .return_class import AbstractApiClass


class CloudflareOauthUrl(AbstractApiClass):
    """
        Cloudflare OAuth authorization URL

        Args:
            client (ApiClient): An authenticated API Client instance
            url (str): Cloudflare's consent page for the Abacus.AI OAuth client
    """

    def __init__(self, client, url=None):
        super().__init__(client, None)
        self.url = url
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'url': repr(self.url)}
        class_name = "CloudflareOauthUrl"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'url': self.url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
