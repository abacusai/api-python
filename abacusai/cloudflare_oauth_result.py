from .return_class import AbstractApiClass


class CloudflareOauthResult(AbstractApiClass):
    """
        Outcome of a Cloudflare OAuth flow once Cloudflare sends the user back

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppDomainId (id): The ID of the web app domain
            hostname (str): The domain that was configured
            returnUrl (str): Where the user started the flow
            success (bool): Whether the records were written
            errorCode (str): A generic failure code when success is false
    """

    def __init__(self, client, webAppDomainId=None, hostname=None, returnUrl=None, success=None, errorCode=None):
        super().__init__(client, None)
        self.web_app_domain_id = webAppDomainId
        self.hostname = hostname
        self.return_url = returnUrl
        self.success = success
        self.error_code = errorCode
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_domain_id': repr(self.web_app_domain_id), f'hostname': repr(self.hostname), f'return_url': repr(
            self.return_url), f'success': repr(self.success), f'error_code': repr(self.error_code)}
        class_name = "CloudflareOauthResult"
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
                'return_url': self.return_url, 'success': self.success, 'error_code': self.error_code}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
