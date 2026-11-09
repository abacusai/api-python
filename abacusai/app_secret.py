from .return_class import AbstractApiClass


class AppSecret(AbstractApiClass):
    """
        An organization admin's app secret: an environment variable for web apps whose production value is write-only.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The environment variable name.
            createdAt (str): When the secret was created.
            updatedAt (str): When the value last changed.
            createdByEmail (str): Who created it.
            updatedByEmail (str): Who last changed a value.
            webAppProjectIds (list): The web app projects it is attached to.
    """

    def __init__(self, client, name=None, createdAt=None, updatedAt=None, createdByEmail=None, updatedByEmail=None, webAppProjectIds=None):
        super().__init__(client, None)
        self.name = name
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.created_by_email = createdByEmail
        self.updated_by_email = updatedByEmail
        self.web_app_project_ids = webAppProjectIds
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'created_by_email': repr(
            self.created_by_email), f'updated_by_email': repr(self.updated_by_email), f'web_app_project_ids': repr(self.web_app_project_ids)}
        class_name = "AppSecret"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'created_at': self.created_at, 'updated_at': self.updated_at, 'created_by_email': self.created_by_email,
                'updated_by_email': self.updated_by_email, 'web_app_project_ids': self.web_app_project_ids}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
