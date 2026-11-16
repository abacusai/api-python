from .return_class import AbstractApiClass


class WebAppHealthSummary(AbstractApiClass):
    """
        Self-healing and security-scan status of a web app project (_getWebAppProjectsHealth).

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppProjectId (id): The project ID.
            selfHealingEnabled (bool): Whether self-healing is turned on for the app.
            selfHealingIntervalHours (int): Hours between self-healing error scans of the app (8, 24 or 168).
            lastSecurityScanAt (str): When the app's last security scan completed. Omitted when it never had one.
            securityScanOnDeploy (bool): Whether the app is scanned after a production deploy. Omitted when scan-on-deploy is unavailable to the org.
            securityScanIntervalHours (int): Hours after a scan before a deploy scans the app again; 0 scans after every deploy. Omitted like security_scan_on_deploy.
    """

    def __init__(self, client, webAppProjectId=None, selfHealingEnabled=None, selfHealingIntervalHours=None, lastSecurityScanAt=None, securityScanOnDeploy=None, securityScanIntervalHours=None):
        super().__init__(client, None)
        self.web_app_project_id = webAppProjectId
        self.self_healing_enabled = selfHealingEnabled
        self.self_healing_interval_hours = selfHealingIntervalHours
        self.last_security_scan_at = lastSecurityScanAt
        self.security_scan_on_deploy = securityScanOnDeploy
        self.security_scan_interval_hours = securityScanIntervalHours
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_project_id': repr(self.web_app_project_id), f'self_healing_enabled': repr(self.self_healing_enabled), f'self_healing_interval_hours': repr(self.self_healing_interval_hours), f'last_security_scan_at': repr(
            self.last_security_scan_at), f'security_scan_on_deploy': repr(self.security_scan_on_deploy), f'security_scan_interval_hours': repr(self.security_scan_interval_hours)}
        class_name = "WebAppHealthSummary"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_project_id': self.web_app_project_id, 'self_healing_enabled': self.self_healing_enabled, 'self_healing_interval_hours': self.self_healing_interval_hours,
                'last_security_scan_at': self.last_security_scan_at, 'security_scan_on_deploy': self.security_scan_on_deploy, 'security_scan_interval_hours': self.security_scan_interval_hours}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
