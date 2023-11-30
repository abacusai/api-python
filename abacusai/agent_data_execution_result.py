from .return_class import AbstractApiClass


class AgentDataExecutionResult(AbstractApiClass):
    """
        Results of agent execution with uploaded data.

        Args:
            client (ApiClient): An authenticated API Client instance
            docIds (list[str]): A list of document IDs uploaded to agent.
            response (str): The result of agent conversation execution.
            deploymentConversationId (id): The unique identifier of the deployment conversation.
    """

    def __init__(self, client, docIds=None, response=None, deploymentConversationId=None):
        super().__init__(client, None)
        self.doc_ids = docIds
        self.response = response
        self.deployment_conversation_id = deploymentConversationId

    def __repr__(self):
        repr_dict = {f'doc_ids': repr(self.doc_ids), f'response': repr(
            self.response), f'deployment_conversation_id': repr(self.deployment_conversation_id)}
        class_name = "AgentDataExecutionResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'doc_ids': self.doc_ids, 'response': self.response,
                'deployment_conversation_id': self.deployment_conversation_id}
        return {key: value for key, value in resp.items() if value is not None}
