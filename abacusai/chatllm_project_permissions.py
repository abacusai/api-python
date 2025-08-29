from .return_class import AbstractApiClass


class ChatllmProjectPermissions(AbstractApiClass):
    """
        ChatLLM Project Permissions

        Args:
            client (ApiClient): An authenticated API Client instance
            chatllmProjectId (id): The ID of the chatllm project.
            accessLevel (str): The access level of the chatllm project.
            userPermissions (list): List of tuples containing (user_id, permission).
            userGroupPermissions (list): List of tuples containing (user_group_id, permission).
    """

    def __init__(self, client, chatllmProjectId=None, accessLevel=None, userPermissions=None, userGroupPermissions=None):
        super().__init__(client, None)
        self.chatllm_project_id = chatllmProjectId
        self.access_level = accessLevel
        self.user_permissions = userPermissions
        self.user_group_permissions = userGroupPermissions
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_project_id': repr(self.chatllm_project_id), f'access_level': repr(
            self.access_level), f'user_permissions': repr(self.user_permissions), f'user_group_permissions': repr(self.user_group_permissions)}
        class_name = "ChatllmProjectPermissions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chatllm_project_id': self.chatllm_project_id, 'access_level': self.access_level,
                'user_permissions': self.user_permissions, 'user_group_permissions': self.user_group_permissions}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
