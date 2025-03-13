from .return_class import AbstractApiClass


class ChatllmProject(AbstractApiClass):
    """
        ChatLLM Project

        Args:
            client (ApiClient): An authenticated API Client instance
            chatllmProjectId (id): The ID of the chatllm project.
            name (str): The name of the chatllm project.
            description (str): The description of the chatllm project.
            customInstructions (str): The custom instructions of the chatllm project.
            createdAt (str): The creation time of the chatllm project.
            updatedAt (str): The update time of the chatllm project.
    """

    def __init__(self, client, chatllmProjectId=None, name=None, description=None, customInstructions=None, createdAt=None, updatedAt=None):
        super().__init__(client, chatllmProjectId)
        self.chatllm_project_id = chatllmProjectId
        self.name = name
        self.description = description
        self.custom_instructions = customInstructions
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_project_id': repr(self.chatllm_project_id), f'name': repr(self.name), f'description': repr(
            self.description), f'custom_instructions': repr(self.custom_instructions), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "ChatllmProject"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chatllm_project_id': self.chatllm_project_id, 'name': self.name, 'description': self.description,
                'custom_instructions': self.custom_instructions, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
