from .return_class import AbstractApiClass


class DaemonTaskConversation(AbstractApiClass):
    """
        Daemon Task Conversation

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (id): The ID of the deployment conversation
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
            deploymentConversationName (str): The name of the conversation
            externalApplicationId (str): The external application ID
    """

    def __init__(self, client, deploymentConversationId=None, createdAt=None, updatedAt=None, deploymentConversationName=None, externalApplicationId=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deployment_conversation_name = deploymentConversationName
        self.external_application_id = externalApplicationId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'created_at': repr(self.created_at), f'updated_at': repr(
            self.updated_at), f'deployment_conversation_name': repr(self.deployment_conversation_name), f'external_application_id': repr(self.external_application_id)}
        class_name = "DaemonTaskConversation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id, 'created_at': self.created_at, 'updated_at': self.updated_at,
                'deployment_conversation_name': self.deployment_conversation_name, 'external_application_id': self.external_application_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
