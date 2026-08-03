from .return_class import AbstractApiClass


class HostedAppContainer(AbstractApiClass):
    """
        Deep agent app listing information.

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (id): The deployment conversation ID
            name (str): The name of the app
            userId (id): The ID of the creation user
            email (str): The email of the creation user
            createdAt (str): Creation timestamp
            updatedAt (str): Last update timestamp
            isDeployable (bool): Can this version be deployed
            deployedStatus (str): Deployment status (PENDING/ACTIVE/STOPPED/NOT_DEPLOYED)
            accessLevel (str): Access Level (PUBLIC/PRIVATE/DEDICATED/OWNER_ONLY)
            hostnames (list[dict]): Hostnames and tags of the deployed app
            llmArtifactId (id): The ID of the LLM artifact
            artifactType (str): The type of the artifact
            deployedLlmArtifactId (id): The ID of the deployed LLM artifact
            hasDatabase (bool): Whether the app has a database associated to it
            hasStorage (bool): Whether the app has a cloud storage associated to it
            webAppProjectId (id): The ID of the web app project
            parentConversationId (id): The ID of the parent conversation
            projectMetadata (dict): The metadata of the web app project
            memoryGb (float): The memory in GB of the web app deployment
            domainAliases (list): The domain aliases of the deployed app
            isThrottled (bool): Whether the app was stopped due to excessive resource usage
            scDeployedUrls (list): Live deployed URLs of a supercomputer conversation (tracked in conversation metadata)
    """

    def __init__(self, client, deploymentConversationId=None, name=None, userId=None, email=None, createdAt=None, updatedAt=None, isDeployable=None, deployedStatus=None, accessLevel=None, hostnames=None, llmArtifactId=None, artifactType=None, deployedLlmArtifactId=None, hasDatabase=None, hasStorage=None, webAppProjectId=None, parentConversationId=None, projectMetadata=None, memoryGb=None, domainAliases=None, isThrottled=None, scDeployedUrls=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.user_id = userId
        self.email = email
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.is_deployable = isDeployable
        self.deployed_status = deployedStatus
        self.access_level = accessLevel
        self.hostnames = hostnames
        self.llm_artifact_id = llmArtifactId
        self.artifact_type = artifactType
        self.deployed_llm_artifact_id = deployedLlmArtifactId
        self.has_database = hasDatabase
        self.has_storage = hasStorage
        self.web_app_project_id = webAppProjectId
        self.parent_conversation_id = parentConversationId
        self.project_metadata = projectMetadata
        self.memory_gb = memoryGb
        self.domain_aliases = domainAliases
        self.is_throttled = isThrottled
        self.sc_deployed_urls = scDeployedUrls
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'name': repr(self.name), f'user_id': repr(self.user_id), f'email': repr(self.email), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'is_deployable': repr(self.is_deployable), f'deployed_status': repr(self.deployed_status), f'access_level': repr(self.access_level), f'hostnames': repr(self.hostnames), f'llm_artifact_id': repr(self.llm_artifact_id), f'artifact_type': repr(
            self.artifact_type), f'deployed_llm_artifact_id': repr(self.deployed_llm_artifact_id), f'has_database': repr(self.has_database), f'has_storage': repr(self.has_storage), f'web_app_project_id': repr(self.web_app_project_id), f'parent_conversation_id': repr(self.parent_conversation_id), f'project_metadata': repr(self.project_metadata), f'memory_gb': repr(self.memory_gb), f'domain_aliases': repr(self.domain_aliases), f'is_throttled': repr(self.is_throttled), f'sc_deployed_urls': repr(self.sc_deployed_urls)}
        class_name = "HostedAppContainer"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'user_id': self.user_id, 'email': self.email, 'created_at': self.created_at, 'updated_at': self.updated_at, 'is_deployable': self.is_deployable, 'deployed_status': self.deployed_status, 'access_level': self.access_level, 'hostnames': self.hostnames, 'llm_artifact_id': self.llm_artifact_id, 'artifact_type': self.artifact_type,
                'deployed_llm_artifact_id': self.deployed_llm_artifact_id, 'has_database': self.has_database, 'has_storage': self.has_storage, 'web_app_project_id': self.web_app_project_id, 'parent_conversation_id': self.parent_conversation_id, 'project_metadata': self.project_metadata, 'memory_gb': self.memory_gb, 'domain_aliases': self.domain_aliases, 'is_throttled': self.is_throttled, 'sc_deployed_urls': self.sc_deployed_urls}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
