from .return_class import AbstractApiClass


class NlpChatResponse(AbstractApiClass):
    """
        A chat response from an LLM

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (str): The unique identifier of the deployment conversation.
            messages (list): The conversation messages in the chat.
    """

    def __init__(self, client, deploymentConversationId=None, messages=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.messages = messages
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'messages': repr(self.messages)}
        class_name = "NlpChatResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id,
                'messages': self.messages}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
