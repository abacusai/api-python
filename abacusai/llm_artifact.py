from .return_class import AbstractApiClass


class LlmArtifact(AbstractApiClass):
    """
        LLM Artifact

        Args:
            client (ApiClient): An authenticated API Client instance
            llmArtifactId (id): The ID of the LLM artifact
            info (dict): The info of the LLM artifact
            description (str): The description of the LLM artifact
            createdAt (str): The creation timestamp
    """

    def __init__(self, client, llmArtifactId=None, info=None, description=None, createdAt=None):
        super().__init__(client, llmArtifactId)
        self.llm_artifact_id = llmArtifactId
        self.info = info
        self.description = description
        self.created_at = createdAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'llm_artifact_id': repr(self.llm_artifact_id), f'info': repr(
            self.info), f'description': repr(self.description), f'created_at': repr(self.created_at)}
        class_name = "LlmArtifact"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'llm_artifact_id': self.llm_artifact_id, 'info': self.info,
                'description': self.description, 'created_at': self.created_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
