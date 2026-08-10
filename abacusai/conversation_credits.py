from .return_class import AbstractApiClass


class ConversationCredits(AbstractApiClass):
    """
        Credits one conversation spent on a single day (_getConversationCreditsForDay).

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (id): The ID of the deployment conversation.
            name (str): The conversation title.
            conversationType (str): The type of the conversation.
            isDeleted (bool): True when the conversation itself has been deleted; its credits survive.
            unit (str): 'credits' for self-serve, 'usd' for enterprise -- the unit `credits` is expressed in.
            credits (float): Spend on the requested day only, not the conversation's lifetime.
            turns (int): Conversation events on that day.
    """

    def __init__(self, client, deploymentConversationId=None, name=None, conversationType=None, isDeleted=None, unit=None, credits=None, turns=None):
        super().__init__(client, None)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.conversation_type = conversationType
        self.is_deleted = isDeleted
        self.unit = unit
        self.credits = credits
        self.turns = turns
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'name': repr(self.name), f'conversation_type': repr(
            self.conversation_type), f'is_deleted': repr(self.is_deleted), f'unit': repr(self.unit), f'credits': repr(self.credits), f'turns': repr(self.turns)}
        class_name = "ConversationCredits"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'conversation_type': self.conversation_type,
                'is_deleted': self.is_deleted, 'unit': self.unit, 'credits': self.credits, 'turns': self.turns}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
