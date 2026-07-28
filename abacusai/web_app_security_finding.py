from .return_class import AbstractApiClass


class WebAppSecurityFinding(AbstractApiClass):
    """
        A security finding detected by a security scan of a deployed web app.

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppSecurityFindingId (id): The finding ID
            category (str): DEPENDENCY | SECRET | AUTH | CONFIG | CODE
            severity (str): LOW | MEDIUM | HIGH | CRITICAL
            title (str): Short title of the finding
            detail (str): Detailed description of the vulnerability
            filePath (str): The affected file, when applicable
            recommendation (str): Suggested remediation
            status (str): OPEN | ACKNOWLEDGED | RESOLVED | WONT_FIX
            lastSeen (str): When the finding was last confirmed by a scan
            createdAt (str): When the finding was first detected
    """

    def __init__(self, client, webAppSecurityFindingId=None, category=None, severity=None, title=None, detail=None, filePath=None, recommendation=None, status=None, lastSeen=None, createdAt=None):
        super().__init__(client, webAppSecurityFindingId)
        self.web_app_security_finding_id = webAppSecurityFindingId
        self.category = category
        self.severity = severity
        self.title = title
        self.detail = detail
        self.file_path = filePath
        self.recommendation = recommendation
        self.status = status
        self.last_seen = lastSeen
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_security_finding_id': repr(self.web_app_security_finding_id), f'category': repr(self.category), f'severity': repr(self.severity), f'title': repr(self.title), f'detail': repr(
            self.detail), f'file_path': repr(self.file_path), f'recommendation': repr(self.recommendation), f'status': repr(self.status), f'last_seen': repr(self.last_seen), f'created_at': repr(self.created_at)}
        class_name = "WebAppSecurityFinding"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_security_finding_id': self.web_app_security_finding_id, 'category': self.category, 'severity': self.severity, 'title': self.title, 'detail': self.detail,
                'file_path': self.file_path, 'recommendation': self.recommendation, 'status': self.status, 'last_seen': self.last_seen, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
