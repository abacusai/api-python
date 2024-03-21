from .return_class import AbstractApiClass


class OrganizationExternalApplicationSettings(AbstractApiClass):
    """
        The External Application Settings for an Organization.

        Args:
            client (ApiClient): An authenticated API Client instance
            logo (str): The logo.
            theme (dict): The theme used for External Applications in this org.
            managedUserService (str): The external service that is managing the user accounts.
    """

    def __init__(self, client, logo=None, theme=None, managedUserService=None):
        super().__init__(client, None)
        self.logo = logo
        self.theme = theme
        self.managed_user_service = managedUserService
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'logo': repr(self.logo), f'theme': repr(
            self.theme), f'managed_user_service': repr(self.managed_user_service)}
        class_name = "OrganizationExternalApplicationSettings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'logo': self.logo, 'theme': self.theme,
                'managed_user_service': self.managed_user_service}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
