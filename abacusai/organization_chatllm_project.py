from .return_class import AbstractApiClass


class OrganizationChatllmProject(AbstractApiClass):
    """
        A ChatLLM project in the organization, as its admins list them (listOrganizationChatllmProjects).

        Args:
            client (ApiClient): An authenticated API Client instance
            chatllmProjectId (id): The project.
            name (str): The project name.
            description (str): The project description.
            createdByUserId (id): The member who created the project.
            createdAt (str): When the project was created.
            accessLevel (str): Who in the organization can open the project.
    """

    def __init__(self, client, chatllmProjectId=None, name=None, description=None, createdByUserId=None, createdAt=None, accessLevel=None):
        super().__init__(client, None)
        self.chatllm_project_id = chatllmProjectId
        self.name = name
        self.description = description
        self.created_by_user_id = createdByUserId
        self.created_at = createdAt
        self.access_level = accessLevel
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_project_id': repr(self.chatllm_project_id), f'name': repr(self.name), f'description': repr(
            self.description), f'created_by_user_id': repr(self.created_by_user_id), f'created_at': repr(self.created_at), f'access_level': repr(self.access_level)}
        class_name = "OrganizationChatllmProject"
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
                'created_by_user_id': self.created_by_user_id, 'created_at': self.created_at, 'access_level': self.access_level}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
