from .agent_data_document_info import AgentDataDocumentInfo
from .return_class import AbstractApiClass


class AgentDataExecutionResult(AbstractApiClass):
    """
        Results of agent execution with uploaded data.

        Args:
            client (ApiClient): An authenticated API Client instance
            response (str): The result of agent conversation execution.
            deploymentConversationId (id): The unique identifier of the deployment conversation.
            docInfos (AgentDataDocumentInfo): A list of dict containing information on documents uploaded to agent.
    """

    def __init__(self, client, response=None, deploymentConversationId=None, docInfos={}):
        super().__init__(client, None)
        self.response = response
        self.deployment_conversation_id = deploymentConversationId
        self.doc_infos = client._build_class(AgentDataDocumentInfo, docInfos)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'response': repr(self.response), f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'doc_infos': repr(self.doc_infos)}
        class_name = "AgentDataExecutionResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'response': self.response, 'deployment_conversation_id': self.deployment_conversation_id,
                'doc_infos': self._get_attribute_as_dict(self.doc_infos)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
