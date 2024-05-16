from .return_class import AbstractApiClass


class LlmApp(AbstractApiClass):
    """
        An LLM App that can be used for generation. LLM Apps are specifically crafted to help with certain tasks like code generation or question answering.

        Args:
            client (ApiClient): An authenticated API Client instance
            llmAppId (str): The unique identifier of the LLM App.
            name (str): The name of the LLM App.
            description (str): The description of the LLM App.
            projectId (str): The project ID of the deployment associated with the LLM App.
            deploymentId (str): The deployment ID associated with the LLM App.
            createdAt (str): The timestamp at which the LLM App was created.
            updatedAt (str): The timestamp at which the LLM App was updated.
            status (str): The status of the LLM App's deployment.
    """

    def __init__(self, client, llmAppId=None, name=None, description=None, projectId=None, deploymentId=None, createdAt=None, updatedAt=None, status=None):
        super().__init__(client, llmAppId)
        self.llm_app_id = llmAppId
        self.name = name
        self.description = description
        self.project_id = projectId
        self.deployment_id = deploymentId
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.status = status
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'llm_app_id': repr(self.llm_app_id), f'name': repr(self.name), f'description': repr(self.description), f'project_id': repr(
            self.project_id), f'deployment_id': repr(self.deployment_id), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'status': repr(self.status)}
        class_name = "LlmApp"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'llm_app_id': self.llm_app_id, 'name': self.name, 'description': self.description, 'project_id': self.project_id,
                'deployment_id': self.deployment_id, 'created_at': self.created_at, 'updated_at': self.updated_at, 'status': self.status}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
