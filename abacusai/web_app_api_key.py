from .return_class import AbstractApiClass


class WebAppApiKey(AbstractApiClass):
    """
        A web app and the LLM API key bound to it.

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppProjectId (id): The ID of the web app project (the logical app).
            apiKeyId (id): The ID of the LLM API key bound to the app.
            hostname (str): A hostname the app is served on.
    """

    def __init__(self, client, webAppProjectId=None, apiKeyId=None, hostname=None):
        super().__init__(client, None)
        self.web_app_project_id = webAppProjectId
        self.api_key_id = apiKeyId
        self.hostname = hostname
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_project_id': repr(self.web_app_project_id), f'api_key_id': repr(
            self.api_key_id), f'hostname': repr(self.hostname)}
        class_name = "WebAppApiKey"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_project_id': self.web_app_project_id,
                'api_key_id': self.api_key_id, 'hostname': self.hostname}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
