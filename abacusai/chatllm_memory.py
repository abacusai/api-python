from .return_class import AbstractApiClass


class ChatllmMemory(AbstractApiClass):
    """
        An LLM created memory in ChatLLM

        Args:
            client (ApiClient): An authenticated API Client instance
            chatllmMemoryId (str): The ID of the chatllm memory.
            memory (str): The text of the ChatLLM memory.
            sourceDeploymentConversationId (str): The deployment conversation where this memory was created.
    """

    def __init__(self, client, chatllmMemoryId=None, memory=None, sourceDeploymentConversationId=None):
        super().__init__(client, chatllmMemoryId)
        self.chatllm_memory_id = chatllmMemoryId
        self.memory = memory
        self.source_deployment_conversation_id = sourceDeploymentConversationId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_memory_id': repr(self.chatllm_memory_id), f'memory': repr(
            self.memory), f'source_deployment_conversation_id': repr(self.source_deployment_conversation_id)}
        class_name = "ChatllmMemory"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chatllm_memory_id': self.chatllm_memory_id, 'memory': self.memory,
                'source_deployment_conversation_id': self.source_deployment_conversation_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
