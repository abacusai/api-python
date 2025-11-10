from .return_class import AbstractApiClass


class HostedAppContainer(AbstractApiClass):
    """
        Hosted app + Deep agent container information.

        Args:
            client (ApiClient): An authenticated API Client instance
            hostedAppContainerId (id): The ID of the hosted app container
            hostedAppId (id): The ID of the hosted app
            deploymentConversationId (id): The deployment conversation ID
            hostedAppVersion (id): The instance of the hosted app
            name (str): The name of the hosted app
            userId (id): The ID of the creation user
            email (str): The email of the creation user
            createdAt (str): Creation timestamp
            updatedAt (str): Last update timestamp
            containerImage (str): Container image name
            route (str): Container route
            appConfig (dict): App configuration
            isDev (bool): Whether this is a dev container
            isDeployable (bool): Can this version be deployed
            isPreviewAvailable (bool): Is the dev preview available on the container
            lifecycle (str): Container lifecycle status (PENDING/DEPLOYING/ACTIVE/FAILED/STOPPED/DELETING)
            status (str): Container status (RUNNING/STOPPED/DEPLOYING/FAILED)
            deployedStatus (str): Deployment status (PENDING/ACTIVE/STOPPED/NOT_DEPLOYED)
            accessLevel (str): Access Level (PUBLIC/PRIVATE/DEDICATED)
            hostname (str): Hostname of the deployed app
            llmArtifactId (id): The ID of the LLM artifact
            artifactType (str): The type of the artifact
            deployedLlmArtifactId (id): The ID of the deployed LLM artifact
            hasDatabase (bool): Whether the app has a database associated to it
            hasStorage (bool): Whether the app has a cloud storage associated to it
            webAppProjectId (id): The ID of the web app project
            parentConversationId (id): The ID of the parent conversation
            projectMetadata (dict): The metadata of the web app project
    """

    def __init__(self, client, hostedAppContainerId=None, hostedAppId=None, deploymentConversationId=None, hostedAppVersion=None, name=None, userId=None, email=None, createdAt=None, updatedAt=None, containerImage=None, route=None, appConfig=None, isDev=None, isDeployable=None, isPreviewAvailable=None, lifecycle=None, status=None, deployedStatus=None, accessLevel=None, hostname=None, llmArtifactId=None, artifactType=None, deployedLlmArtifactId=None, hasDatabase=None, hasStorage=None, webAppProjectId=None, parentConversationId=None, projectMetadata=None):
        super().__init__(client, hostedAppContainerId)
        self.hosted_app_container_id = hostedAppContainerId
        self.hosted_app_id = hostedAppId
        self.deployment_conversation_id = deploymentConversationId
        self.hosted_app_version = hostedAppVersion
        self.name = name
        self.user_id = userId
        self.email = email
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.container_image = containerImage
        self.route = route
        self.app_config = appConfig
        self.is_dev = isDev
        self.is_deployable = isDeployable
        self.is_preview_available = isPreviewAvailable
        self.lifecycle = lifecycle
        self.status = status
        self.deployed_status = deployedStatus
        self.access_level = accessLevel
        self.hostname = hostname
        self.llm_artifact_id = llmArtifactId
        self.artifact_type = artifactType
        self.deployed_llm_artifact_id = deployedLlmArtifactId
        self.has_database = hasDatabase
        self.has_storage = hasStorage
        self.web_app_project_id = webAppProjectId
        self.parent_conversation_id = parentConversationId
        self.project_metadata = projectMetadata
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'hosted_app_container_id': repr(self.hosted_app_container_id), f'hosted_app_id': repr(self.hosted_app_id), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'hosted_app_version': repr(self.hosted_app_version), f'name': repr(self.name), f'user_id': repr(self.user_id), f'email': repr(self.email), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'container_image': repr(self.container_image), f'route': repr(self.route), f'app_config': repr(self.app_config), f'is_dev': repr(self.is_dev), f'is_deployable': repr(self.is_deployable), f'is_preview_available': repr(
            self.is_preview_available), f'lifecycle': repr(self.lifecycle), f'status': repr(self.status), f'deployed_status': repr(self.deployed_status), f'access_level': repr(self.access_level), f'hostname': repr(self.hostname), f'llm_artifact_id': repr(self.llm_artifact_id), f'artifact_type': repr(self.artifact_type), f'deployed_llm_artifact_id': repr(self.deployed_llm_artifact_id), f'has_database': repr(self.has_database), f'has_storage': repr(self.has_storage), f'web_app_project_id': repr(self.web_app_project_id), f'parent_conversation_id': repr(self.parent_conversation_id), f'project_metadata': repr(self.project_metadata)}
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
        resp = {'hosted_app_container_id': self.hosted_app_container_id, 'hosted_app_id': self.hosted_app_id, 'deployment_conversation_id': self.deployment_conversation_id, 'hosted_app_version': self.hosted_app_version, 'name': self.name, 'user_id': self.user_id, 'email': self.email, 'created_at': self.created_at, 'updated_at': self.updated_at, 'container_image': self.container_image, 'route': self.route, 'app_config': self.app_config, 'is_dev': self.is_dev, 'is_deployable': self.is_deployable, 'is_preview_available': self.is_preview_available,
                'lifecycle': self.lifecycle, 'status': self.status, 'deployed_status': self.deployed_status, 'access_level': self.access_level, 'hostname': self.hostname, 'llm_artifact_id': self.llm_artifact_id, 'artifact_type': self.artifact_type, 'deployed_llm_artifact_id': self.deployed_llm_artifact_id, 'has_database': self.has_database, 'has_storage': self.has_storage, 'web_app_project_id': self.web_app_project_id, 'parent_conversation_id': self.parent_conversation_id, 'project_metadata': self.project_metadata}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
