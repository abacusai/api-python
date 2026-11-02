from .return_class import AbstractApiClass


class CloudflareOauthState(AbstractApiClass):
    """
        Where a Cloudflare OAuth flow started, read from the signed state Cloudflare sent back

        Args:
            client (ApiClient): An authenticated API Client instance
            returnOrigin (str): The origin (scheme and host) of the page the flow started on
    """

    def __init__(self, client, returnOrigin=None):
        super().__init__(client, None)
        self.return_origin = returnOrigin
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'return_origin': repr(self.return_origin)}
        class_name = "CloudflareOauthState"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'return_origin': self.return_origin}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
