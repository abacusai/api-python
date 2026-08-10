from .return_class import AbstractApiClass


class Vertical(AbstractApiClass):
    """
        A ChatLLM Vertical (e.g., Health).

        Args:
            client (ApiClient): An authenticated API Client instance
            externalApplicationId (str): The ID of the vertical's external application.
            deploymentId (str): The ID of the vertical's deployment.
            name (str): The name of the vertical.
            description (str): The description of the vertical.
            verticalType (str): The type of vertical (e.g., HEALTH).
            urlPath (str): The vertical's routing slug (e.g. 'health', or 'legal-agent' for an Agent app).
            verticalAgentMode (bool): Whether this is the vertical's Agent app rather than its Chat app. Legal and Finance expose both on one deployment, so they share a vertical_type and this is what tells them apart.
    """

    def __init__(self, client, externalApplicationId=None, deploymentId=None, name=None, description=None, verticalType=None, urlPath=None, verticalAgentMode=None):
        super().__init__(client, None)
        self.external_application_id = externalApplicationId
        self.deployment_id = deploymentId
        self.name = name
        self.description = description
        self.vertical_type = verticalType
        self.url_path = urlPath
        self.vertical_agent_mode = verticalAgentMode
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'external_application_id': repr(self.external_application_id), f'deployment_id': repr(self.deployment_id), f'name': repr(self.name), f'description': repr(
            self.description), f'vertical_type': repr(self.vertical_type), f'url_path': repr(self.url_path), f'vertical_agent_mode': repr(self.vertical_agent_mode)}
        class_name = "Vertical"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'external_application_id': self.external_application_id, 'deployment_id': self.deployment_id, 'name': self.name,
                'description': self.description, 'vertical_type': self.vertical_type, 'url_path': self.url_path, 'vertical_agent_mode': self.vertical_agent_mode}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
