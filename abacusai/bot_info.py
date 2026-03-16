from .external_application import ExternalApplication
from .return_class import AbstractApiClass


class BotInfo(AbstractApiClass):
    """
        Information about an external application and LLM.

        Args:
            client (ApiClient): An authenticated API Client instance
            externalApplicationId (str): The external application ID.
            deploymentId (str): The deployment ID.
            externalApplication (ExternalApplication): The described external application details.
    """

    def __init__(self, client, externalApplicationId=None, deploymentId=None, externalApplication={}):
        super().__init__(client, None)
        self.external_application_id = externalApplicationId
        self.deployment_id = deploymentId
        self.external_application = client._build_class(
            ExternalApplication, externalApplication)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'external_application_id': repr(self.external_application_id), f'deployment_id': repr(
            self.deployment_id), f'external_application': repr(self.external_application)}
        class_name = "BotInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'external_application_id': self.external_application_id, 'deployment_id': self.deployment_id,
                'external_application': self._get_attribute_as_dict(self.external_application)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
