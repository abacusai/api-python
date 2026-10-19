from .hosted_database_finding import HostedDatabaseFinding
from .return_class import AbstractApiClass


class HostedDatabaseHealth(AbstractApiClass):
    """
        Database health of a web app's database (_getHostedDatabaseHealth): the findings recorded within the last day.

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppProjectId (id): The project ID
            hostedDatabaseId (id): The database examined
            databaseName (str): The database's display name
            deploymentConversationId (id): The app's build conversation ID
            canFix (bool): Whether the caller may send findings to the agent (the app owner or an admin)
            scannedAt (str): When the findings were last recorded
            findings (HostedDatabaseFinding): The findings seen within the last day, most recently seen first
    """

    def __init__(self, client, webAppProjectId=None, hostedDatabaseId=None, databaseName=None, deploymentConversationId=None, canFix=None, scannedAt=None, findings={}):
        super().__init__(client, None)
        self.web_app_project_id = webAppProjectId
        self.hosted_database_id = hostedDatabaseId
        self.database_name = databaseName
        self.deployment_conversation_id = deploymentConversationId
        self.can_fix = canFix
        self.scanned_at = scannedAt
        self.findings = client._build_class(HostedDatabaseFinding, findings)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_project_id': repr(self.web_app_project_id), f'hosted_database_id': repr(self.hosted_database_id), f'database_name': repr(
            self.database_name), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'can_fix': repr(self.can_fix), f'scanned_at': repr(self.scanned_at), f'findings': repr(self.findings)}
        class_name = "HostedDatabaseHealth"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_project_id': self.web_app_project_id, 'hosted_database_id': self.hosted_database_id, 'database_name': self.database_name,
                'deployment_conversation_id': self.deployment_conversation_id, 'can_fix': self.can_fix, 'scanned_at': self.scanned_at, 'findings': self._get_attribute_as_dict(self.findings)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
