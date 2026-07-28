from .return_class import AbstractApiClass
from .web_app_issue import WebAppIssue


class WebAppSelfHealInfo(AbstractApiClass):
    """
        Self-heal issues and settings for a web app project (_listWebAppIssues).

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppProjectId (id): The project ID
            appName (str): Name of the app's build conversation
            deploymentConversationId (id): The app's build conversation ID
            stagingHostname (str): The hostname fixes are staged on
            projectType (str): web_app | web_service | mobile_app
            settings (dict): The project's self_heal settings (enabled, notify_recipients, tag, hostname, capture_client_errors)
            hasRecentTraffic (bool): Whether a deployment served a successful external request recently
            issues (WebAppIssue): The project's issues
    """

    def __init__(self, client, webAppProjectId=None, appName=None, deploymentConversationId=None, stagingHostname=None, projectType=None, settings=None, hasRecentTraffic=None, issues={}):
        super().__init__(client, None)
        self.web_app_project_id = webAppProjectId
        self.app_name = appName
        self.deployment_conversation_id = deploymentConversationId
        self.staging_hostname = stagingHostname
        self.project_type = projectType
        self.settings = settings
        self.has_recent_traffic = hasRecentTraffic
        self.issues = client._build_class(WebAppIssue, issues)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_project_id': repr(self.web_app_project_id), f'app_name': repr(self.app_name), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'staging_hostname': repr(
            self.staging_hostname), f'project_type': repr(self.project_type), f'settings': repr(self.settings), f'has_recent_traffic': repr(self.has_recent_traffic), f'issues': repr(self.issues)}
        class_name = "WebAppSelfHealInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_project_id': self.web_app_project_id, 'app_name': self.app_name, 'deployment_conversation_id': self.deployment_conversation_id, 'staging_hostname': self.staging_hostname,
                'project_type': self.project_type, 'settings': self.settings, 'has_recent_traffic': self.has_recent_traffic, 'issues': self._get_attribute_as_dict(self.issues)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
