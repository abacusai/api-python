from .api_class import AgentConversationMessage
from .return_class import AbstractApiClass


class AgentConversation(AbstractApiClass):
    """
        List of messages with Agent chat

        Args:
            client (ApiClient): An authenticated API Client instance
            messages (AgentConversationMessage): list of messages in the conversation with agent.
    """

    def __init__(self, client, messages={}):
        super().__init__(client, None)
        self.messages = client._build_class(AgentConversationMessage, messages)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'messages': repr(self.messages)}
        class_name = "AgentConversation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'messages': self._get_attribute_as_dict(self.messages)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
