from .return_class import AbstractApiClass
from .web_app_security_finding import WebAppSecurityFinding


class WebAppSecurityScanInfo(AbstractApiClass):
    """
        Security scan state and findings for a web app project (_listAppSecurityFindings).

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppProjectId (id): The project ID
            appName (str): Name of the app's build conversation
            deploymentConversationId (id): The app's build conversation ID
            isDeployed (bool): Whether the app currently has an active deployment
            scanStatus (str): Status of the latest scan (RUNNING | COMPLETE | FAILED), if any
            scanStartedAt (str): When the latest scan started
            scanCompletedAt (str): When the latest scan completed
            scanError (str): Error message when the latest scan failed
            settings (dict): The project's security_scan settings (notify_recipients)
            findings (WebAppSecurityFinding): The project's findings
    """

    def __init__(self, client, webAppProjectId=None, appName=None, deploymentConversationId=None, isDeployed=None, scanStatus=None, scanStartedAt=None, scanCompletedAt=None, scanError=None, settings=None, findings={}):
        super().__init__(client, None)
        self.web_app_project_id = webAppProjectId
        self.app_name = appName
        self.deployment_conversation_id = deploymentConversationId
        self.is_deployed = isDeployed
        self.scan_status = scanStatus
        self.scan_started_at = scanStartedAt
        self.scan_completed_at = scanCompletedAt
        self.scan_error = scanError
        self.settings = settings
        self.findings = client._build_class(WebAppSecurityFinding, findings)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_project_id': repr(self.web_app_project_id), f'app_name': repr(self.app_name), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'is_deployed': repr(self.is_deployed), f'scan_status': repr(
            self.scan_status), f'scan_started_at': repr(self.scan_started_at), f'scan_completed_at': repr(self.scan_completed_at), f'scan_error': repr(self.scan_error), f'settings': repr(self.settings), f'findings': repr(self.findings)}
        class_name = "WebAppSecurityScanInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_project_id': self.web_app_project_id, 'app_name': self.app_name, 'deployment_conversation_id': self.deployment_conversation_id, 'is_deployed': self.is_deployed, 'scan_status': self.scan_status,
                'scan_started_at': self.scan_started_at, 'scan_completed_at': self.scan_completed_at, 'scan_error': self.scan_error, 'settings': self.settings, 'findings': self._get_attribute_as_dict(self.findings)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
