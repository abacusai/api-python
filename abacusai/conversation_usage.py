from .return_class import AbstractApiClass


class ConversationUsage(AbstractApiClass):
    """
        One conversation's spend on one UTC day (listConversationUsage).

        Args:
            client (ApiClient): An authenticated API Client instance
            billDate (str): The UTC day, YYYY-MM-DD.
            deploymentConversationId (id): The conversation.
            name (str): The conversation title.
            conversationType (str): The type of the conversation.
            chatllmProjectId (id): The ChatLLM project the conversation is in. Absent when it is in none, or has been deleted.
            chatllmProjectName (str): That project's name. Absent when the project no longer exists.
            ownerUserId (id): The member who owns the conversation.
            isDeleted (bool): True when the conversation has been deleted; its spend survives.
            unit (str): 'usd' for enterprise organizations, 'credits' for self-serve: the unit `spend` is in.
            spend (float): Spend on that day. An estimate derived from conversation activity, not an invoice line.
            turns (int): Conversation events on that day.
    """

    def __init__(self, client, billDate=None, deploymentConversationId=None, name=None, conversationType=None, chatllmProjectId=None, chatllmProjectName=None, ownerUserId=None, isDeleted=None, unit=None, spend=None, turns=None):
        super().__init__(client, None)
        self.bill_date = billDate
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.conversation_type = conversationType
        self.chatllm_project_id = chatllmProjectId
        self.chatllm_project_name = chatllmProjectName
        self.owner_user_id = ownerUserId
        self.is_deleted = isDeleted
        self.unit = unit
        self.spend = spend
        self.turns = turns
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'bill_date': repr(self.bill_date), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'name': repr(self.name), f'conversation_type': repr(self.conversation_type), f'chatllm_project_id': repr(
            self.chatllm_project_id), f'chatllm_project_name': repr(self.chatllm_project_name), f'owner_user_id': repr(self.owner_user_id), f'is_deleted': repr(self.is_deleted), f'unit': repr(self.unit), f'spend': repr(self.spend), f'turns': repr(self.turns)}
        class_name = "ConversationUsage"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'bill_date': self.bill_date, 'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'conversation_type': self.conversation_type, 'chatllm_project_id': self.chatllm_project_id,
                'chatllm_project_name': self.chatllm_project_name, 'owner_user_id': self.owner_user_id, 'is_deleted': self.is_deleted, 'unit': self.unit, 'spend': self.spend, 'turns': self.turns}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
