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
            history (DeploymentConversationEvent): The history of the deployment conversation.
    """

    def __init__(self, client, deploymentConversationId=None, name=None, deploymentId=None, createdAt=None, history={}):
        super().__init__(client, deploymentConversationId)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.deployment_id = deploymentId
        self.created_at = createdAt
        self.history = client._build_class(
            DeploymentConversationEvent, history)

    def __repr__(self):
        return f"DeploymentConversation(deployment_conversation_id={repr(self.deployment_conversation_id)},\n  name={repr(self.name)},\n  deployment_id={repr(self.deployment_id)},\n  created_at={repr(self.created_at)},\n  history={repr(self.history)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'deployment_id': self.deployment_id, 'created_at': self.created_at, 'history': self._get_attribute_as_dict(self.history)}

    def get(self):
        """
        Gets a deployment conversation.

        Args:
            deployment_conversation_id (str): Unique ID of the conversation.

        Returns:
            DeploymentConversation: The deployment conversation.
        """
        return self.client.get_deployment_conversation(self.deployment_conversation_id)

    def delete(self):
        """
        Delete a Deployment Conversation.

        Args:
            deployment_conversation_id (str): A unique string identifier associated with the deployment conversation.
        """
        return self.client.delete_deployment_conversation(self.deployment_conversation_id)