from .return_class import AbstractApiClass


class OrganizationExternalApplicationSettings(AbstractApiClass):
    """
        The External Application Settings for an Organization.

        Args:
            client (ApiClient): An authenticated API Client instance
            logo (str): The logo.
            theme (dict): The theme used for External Applications in this org.
    """

    def __init__(self, client, logo=None, theme=None):
        super().__init__(client, None)
        self.logo = logo
        self.theme = theme

    def __repr__(self):
        repr_dict = {f'logo': repr(self.logo), f'theme': repr(self.theme)}
        class_name = "OrganizationExternalApplicationSettings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'logo': self.logo, 'theme': self.theme}
        return {key: value for key, value in resp.items() if value is not None}
