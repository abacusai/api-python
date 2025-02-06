from .return_class import AbstractApiClass


class CodeAgentResponse(AbstractApiClass):
    """
        A response from a Code Agent

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (str): The unique identifier of the deployment conversation.
            messages (list): The conversation messages in the chat.
            toolUseRequest (dict): A request to use an external tool. Contains: - id (str): Unique identifier for the tool use request - input (dict): Input parameters for the tool, e.g. {'command': 'ls'} - name (str): Name of the tool being used, e.g. 'bash' - type (str): Always 'tool_use' to identify this as a tool request
    """

    def __init__(self, client, deploymentConversationId=None, messages=None, toolUseRequest=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.messages = messages
        self.tool_use_request = toolUseRequest
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'messages': repr(
            self.messages), f'tool_use_request': repr(self.tool_use_request)}
        class_name = "CodeAgentResponse"
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
                'messages': self.messages, 'tool_use_request': self.tool_use_request}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
