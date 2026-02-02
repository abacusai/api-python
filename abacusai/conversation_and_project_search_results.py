from .return_class import AbstractApiClass


class ConversationAndProjectSearchResults(AbstractApiClass):
    """
        A conversation and project search results.

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (str): The unique identifier of the deployment conversation.
            name (str): The name of the deployment conversation.
            chatllmProjectId (str): The chatllm project id associated with the deployment conversation.
            chatllmProjectName (str): The name of the chatllm project associated with the deployment conversation.
            createdAt (str): The timestamp at which the deployment conversation was created.
            lastEventCreatedAt (str): The timestamp at which the most recent corresponding deployment conversation event was created at.
            conversationType (str): The type of the conversation, which depicts the application it caters to.
            deploymentId (str): The deployment id associated with the deployment conversation.
            externalApplicationId (str): The external application id associated with the deployment conversation.
    """

    def __init__(self, client, deploymentConversationId=None, name=None, chatllmProjectId=None, chatllmProjectName=None, createdAt=None, lastEventCreatedAt=None, conversationType=None, deploymentId=None, externalApplicationId=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.chatllm_project_id = chatllmProjectId
        self.chatllm_project_name = chatllmProjectName
        self.created_at = createdAt
        self.last_event_created_at = lastEventCreatedAt
        self.conversation_type = conversationType
        self.deployment_id = deploymentId
        self.external_application_id = externalApplicationId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'name': repr(self.name), f'chatllm_project_id': repr(self.chatllm_project_id), f'chatllm_project_name': repr(self.chatllm_project_name), f'created_at': repr(
            self.created_at), f'last_event_created_at': repr(self.last_event_created_at), f'conversation_type': repr(self.conversation_type), f'deployment_id': repr(self.deployment_id), f'external_application_id': repr(self.external_application_id)}
        class_name = "ConversationAndProjectSearchResults"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'chatllm_project_id': self.chatllm_project_id, 'chatllm_project_name': self.chatllm_project_name, 'created_at': self.created_at,
                'last_event_created_at': self.last_event_created_at, 'conversation_type': self.conversation_type, 'deployment_id': self.deployment_id, 'external_application_id': self.external_application_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
