from .deployment_conversation_event import DeploymentConversationEvent
from .return_class import AbstractApiClass


class DeploymentConversation(AbstractApiClass):
    """
        A deployment conversation.

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (str): The unique identifier of the deployment conversation.
            name (str): The name of the deployment conversation.
            deploymentId (str): The deployment id associated with the deployment conversation.
            createdAt (str): The timestamp at which the deployment conversation was created.
            externalSessionId (str): The external session id associated with the deployment conversation.
            history (DeploymentConversationEvent): The history of the deployment conversation.
    """

    def __init__(self, client, deploymentConversationId=None, name=None, deploymentId=None, createdAt=None, externalSessionId=None, history={}):
        super().__init__(client, deploymentConversationId)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.deployment_id = deploymentId
        self.created_at = createdAt
        self.external_session_id = externalSessionId
        self.history = client._build_class(
            DeploymentConversationEvent, history)

    def __repr__(self):
        return f"DeploymentConversation(deployment_conversation_id={repr(self.deployment_conversation_id)},\n  name={repr(self.name)},\n  deployment_id={repr(self.deployment_id)},\n  created_at={repr(self.created_at)},\n  external_session_id={repr(self.external_session_id)},\n  history={repr(self.history)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'deployment_id': self.deployment_id, 'created_at': self.created_at, 'external_session_id': self.external_session_id, 'history': self._get_attribute_as_dict(self.history)}

    def get(self, external_session_id: str = None, deployment_id: str = None):
        """
        Gets a deployment conversation.

        Args:
            external_session_id (str): (Optional) External session ID of the conversation.
            deployment_id (str): (Optional) The deployment this conversation belongs to.

        Returns:
            DeploymentConversation: The deployment conversation.
        """
        return self.client.get_deployment_conversation(self.deployment_conversation_id, external_session_id, deployment_id)

    def delete(self, deployment_id: str = None):
        """
        Delete a Deployment Conversation.

        Args:
            deployment_id (str): (Optional) The deployment this conversation belongs to.
        """
        return self.client.delete_deployment_conversation(self.deployment_conversation_id, deployment_id)

    def set_feedback(self, message_index: int, is_useful: bool = None, is_not_useful: bool = None, feedback: str = None):
        """
        Sets a deployment conversation message as useful or not useful

        Args:
            message_index (int): The index of the deployment conversation message
            is_useful (bool): If the message is useful. If true, the message is useful. If false, clear the useful flag.
            is_not_useful (bool): If the message is not useful. If true, the message is not useful. If set to false, clear the useful flag.
            feedback (str): Optional feedback on why the message is useful or not useful
        """
        return self.client.set_deployment_conversation_feedback(self.deployment_conversation_id, message_index, is_useful, is_not_useful, feedback)

    def rename(self, name: str, deployment_id: str = None):
        """
        Rename a Deployment Conversation.

        Args:
            name (str): The new name of the conversation.
            deployment_id (str): (Optional) The deployment this conversation belongs to.
        """
        return self.client.rename_deployment_conversation(self.deployment_conversation_id, name, deployment_id)
