from .return_class import AbstractApiClass


class WebAppDeployment(AbstractApiClass):
    """
        Web app deployment.

        Args:
            client (ApiClient): An authenticated API Client instance
            webAppDeploymentId (id): The ID of the web app deployment.
            hostname (str): The hostname of the web app deployment.
            accessLevel (str): The access level of the web app deployment.
            llmArtifactId (id): The ID of the LLM artifact.
            artifactsPath (str): The path to the artifacts of the web app deployment.
            applicationType (str): The type of application.
            memoryGb (float): The memory in GB of the web app deployment.
    """

    def __init__(self, client, webAppDeploymentId=None, hostname=None, accessLevel=None, llmArtifactId=None, artifactsPath=None, applicationType=None, memoryGb=None):
        super().__init__(client, webAppDeploymentId)
        self.web_app_deployment_id = webAppDeploymentId
        self.hostname = hostname
        self.access_level = accessLevel
        self.llm_artifact_id = llmArtifactId
        self.artifacts_path = artifactsPath
        self.application_type = applicationType
        self.memory_gb = memoryGb
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'web_app_deployment_id': repr(self.web_app_deployment_id), f'hostname': repr(self.hostname), f'access_level': repr(self.access_level), f'llm_artifact_id': repr(
            self.llm_artifact_id), f'artifacts_path': repr(self.artifacts_path), f'application_type': repr(self.application_type), f'memory_gb': repr(self.memory_gb)}
        class_name = "WebAppDeployment"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'web_app_deployment_id': self.web_app_deployment_id, 'hostname': self.hostname, 'access_level': self.access_level,
                'llm_artifact_id': self.llm_artifact_id, 'artifacts_path': self.artifacts_path, 'application_type': self.application_type, 'memory_gb': self.memory_gb}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
