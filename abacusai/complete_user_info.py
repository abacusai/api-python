from .external_application import ExternalApplication
from .organization_external_application_settings import OrganizationExternalApplicationSettings
from .return_class import AbstractApiClass


class CompleteUserInfo(AbstractApiClass):
    """
        The ChatLLM Teams page-load bundle: the payloads of several page-load endpoints in one

        Args:
            client (ApiClient): An authenticated API Client instance
            validApplicationConnectors (dict): The _listValidApplicationConnectors payload.
            validAgentConnectors (dict): The _listValidAgentConnectors payload.
            availableExperiments (list): The _getAvailableExperiments payload.
            userInfo (InternalUserInfo): The _getUserInfo payload.
            orgAppSettings (OrganizationExternalApplicationSettings): The _describeOrganizationExternalApplicationSettings payload.
            externalApplications (ExternalApplication): The listExternalApplications payload.
            externalApplicationDescribes (ExternalApplication): Full describes of the apps a page load can open (super-agent, RouteLLM base, last-selected bot).
            autoTopupConfig (AutoTopupConfig): The _getAutoTopupConfig payload (org admins only).
    """

    def __init__(self, client, validApplicationConnectors=None, validAgentConnectors=None, availableExperiments=None, userInfo={}, orgAppSettings={}, externalApplications={}, externalApplicationDescribes={}, autoTopupConfig={}):
        super().__init__(client, None)
        self.valid_application_connectors = validApplicationConnectors
        self.valid_agent_connectors = validAgentConnectors
        self.available_experiments = availableExperiments
        self.user_info = client._build_class(InternalUserInfo, userInfo)
        self.org_app_settings = client._build_class(
            OrganizationExternalApplicationSettings, orgAppSettings)
        self.external_applications = client._build_class(
            ExternalApplication, externalApplications)
        self.external_application_describes = client._build_class(
            ExternalApplication, externalApplicationDescribes)
        self.auto_topup_config = client._build_class(
            AutoTopupConfig, autoTopupConfig)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'valid_application_connectors': repr(self.valid_application_connectors), f'valid_agent_connectors': repr(self.valid_agent_connectors), f'available_experiments': repr(self.available_experiments), f'user_info': repr(
            self.user_info), f'org_app_settings': repr(self.org_app_settings), f'external_applications': repr(self.external_applications), f'external_application_describes': repr(self.external_application_describes), f'auto_topup_config': repr(self.auto_topup_config)}
        class_name = "CompleteUserInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'valid_application_connectors': self.valid_application_connectors, 'valid_agent_connectors': self.valid_agent_connectors, 'available_experiments': self.available_experiments, 'user_info': self._get_attribute_as_dict(self.user_info), 'org_app_settings': self._get_attribute_as_dict(
            self.org_app_settings), 'external_applications': self._get_attribute_as_dict(self.external_applications), 'external_application_describes': self._get_attribute_as_dict(self.external_application_describes), 'auto_topup_config': self._get_attribute_as_dict(self.auto_topup_config)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
