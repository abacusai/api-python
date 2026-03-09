from .email_domain import EmailDomain
from .return_class import AbstractApiClass


class WebAppProject(AbstractApiClass):
    """
        Web app project (_describeWebAppProject).

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppProjectId (id): The project ID
            sourceDeploymentConversationId (id): Source deployment conversation ID
            deployedStatus (str): ACTIVE | NOT_DEPLOYED
            hostnames (list): List of hostnames to which the app is deployed
            projectMetadata (dict): Project metadata
            projectType (str): web_app | web_service | mobile_app
            hostedDatabaseId (id): The ID of the hosted database
            hostedDevDatabaseId (id): The ID of the hosted dev database
            emailDomains (EmailDomain): List of EmailDomain (custom domains and email status for this project)
    """

    def __init__(self, client, webAppProjectId=None, sourceDeploymentConversationId=None, deployedStatus=None, hostnames=None, projectMetadata=None, projectType=None, hostedDatabaseId=None, hostedDevDatabaseId=None, emailDomains={}):
        super().__init__(client, webAppProjectId)
        self.web_app_project_id = webAppProjectId
        self.source_deployment_conversation_id = sourceDeploymentConversationId
        self.deployed_status = deployedStatus
        self.hostnames = hostnames
        self.project_metadata = projectMetadata
        self.project_type = projectType
        self.hosted_database_id = hostedDatabaseId
        self.hosted_dev_database_id = hostedDevDatabaseId
        self.email_domains = client._build_class(EmailDomain, emailDomains)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_project_id': repr(self.web_app_project_id), f'source_deployment_conversation_id': repr(self.source_deployment_conversation_id), f'deployed_status': repr(self.deployed_status), f'hostnames': repr(self.hostnames), f'project_metadata': repr(
            self.project_metadata), f'project_type': repr(self.project_type), f'hosted_database_id': repr(self.hosted_database_id), f'hosted_dev_database_id': repr(self.hosted_dev_database_id), f'email_domains': repr(self.email_domains)}
        class_name = "WebAppProject"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_project_id': self.web_app_project_id, 'source_deployment_conversation_id': self.source_deployment_conversation_id, 'deployed_status': self.deployed_status, 'hostnames': self.hostnames, 'project_metadata': self.project_metadata,
                'project_type': self.project_type, 'hosted_database_id': self.hosted_database_id, 'hosted_dev_database_id': self.hosted_dev_database_id, 'email_domains': self._get_attribute_as_dict(self.email_domains)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
