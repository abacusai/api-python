from .return_class import AbstractApiClass


class WebAppDevelopmentCost(AbstractApiClass):
    """
        Rough estimated build cost for a single web app project (_getWebAppProjectDevelopmentCost).

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppProjectId (id): The project ID.
            appCreationCost (float): Rough estimated $ cost to build/iterate this app. Omitted from the response (the filter strips None) when the feature is disabled for the org.
    """

    def __init__(self, client, webAppProjectId=None, appCreationCost=None):
        super().__init__(client, None)
        self.web_app_project_id = webAppProjectId
        self.app_creation_cost = appCreationCost
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_project_id': repr(
            self.web_app_project_id), f'app_creation_cost': repr(self.app_creation_cost)}
        class_name = "WebAppDevelopmentCost"
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
                'app_creation_cost': self.app_creation_cost}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
