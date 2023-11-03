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
            regenerateAttempt (int): The sequence number of regeneration. Not regenerated if 0.
            history (DeploymentConversationEvent): The history of the deployment conversation.
    """

    def __init__(self, client, deploymentConversationId=None, name=None, deploymentId=None, createdAt=None, externalSessionId=None, regenerateAttempt=None, history={}):
        super().__init__(client, deploymentConversationId)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.deployment_id = deploymentId
        self.created_at = createdAt
        self.external_session_id = externalSessionId
        self.regenerate_attempt = regenerateAttempt
        self.history = client._build_class(
            DeploymentConversationEvent, history)

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'name': repr(self.name), f'deployment_id': repr(self.deployment_id), f'created_at': repr(
            self.created_at), f'external_session_id': repr(self.external_session_id), f'regenerate_attempt': repr(self.regenerate_attempt), f'history': repr(self.history)}
        class_name = "DeploymentConversation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'deployment_id': self.deployment_id, 'created_at': self.created_at,
                'external_session_id': self.external_session_id, 'regenerate_attempt': self.regenerate_attempt, 'history': self._get_attribute_as_dict(self.history)}
        return {key: value for key, value in resp.items() if value is not None}

    def get(self, external_session_id: str = None, deployment_id: str = None, deployment_token: str = None):
        """
        Gets a deployment conversation.

        Args:
            external_session_id (str): External session ID of the conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            deployment_token (str): The deployment token to authenticate access to the deployment. This is required if not logged in.

        Returns:
            DeploymentConversation: The deployment conversation.
        """
        return self.client.get_deployment_conversation(self.deployment_conversation_id, external_session_id, deployment_id, deployment_token)

    def delete(self, deployment_id: str = None, deployment_token: str = None):
        """
        Delete a Deployment Conversation.

        Args:
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            deployment_token (str): The deployment token to authenticate access to the deployment. This is required if not logged in.
        """
        return self.client.delete_deployment_conversation(self.deployment_conversation_id, deployment_id, deployment_token)

    def clear(self, external_session_id: str = None, deployment_id: str = None, deployment_token: str = None, user_message_indices: list = None):
        """
        Clear the message history of a Deployment Conversation.

        Args:
            external_session_id (str): The external session id associated with the deployment conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            deployment_token (str): The deployment token to authenticate access to the deployment. This is required if not logged in.
            user_message_indices (list): Optional list of user message indices to clear. The associated bot response will also be cleared. If not provided, all messages will be cleared.
        """
        return self.client.clear_deployment_conversation(self.deployment_conversation_id, external_session_id, deployment_id, deployment_token, user_message_indices)

    def set_feedback(self, message_index: int, is_useful: bool = None, is_not_useful: bool = None, feedback: str = None, deployment_id: str = None, deployment_token: str = None):
        """
        Sets a deployment conversation message as useful or not useful

        Args:
            message_index (int): The index of the deployment conversation message
            is_useful (bool): If the message is useful. If true, the message is useful. If false, clear the useful flag.
            is_not_useful (bool): If the message is not useful. If true, the message is not useful. If set to false, clear the useful flag.
            feedback (str): Optional feedback on why the message is useful or not useful
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            deployment_token (str): The deployment token to authenticate access to the deployment. This is required if not logged in.
        """
        return self.client.set_deployment_conversation_feedback(self.deployment_conversation_id, message_index, is_useful, is_not_useful, feedback, deployment_id, deployment_token)

    def rename(self, name: str, deployment_id: str = None, deployment_token: str = None):
        """
        Rename a Deployment Conversation.

        Args:
            name (str): The new name of the conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            deployment_token (str): The deployment token to authenticate access to the deployment. This is required if not logged in.
        """
        return self.client.rename_deployment_conversation(self.deployment_conversation_id, name, deployment_id, deployment_token)

    def export(self, external_session_id: str = None):
        """
        Export a Deployment Conversation.

        Args:
            external_session_id (str): The external session id associated with the deployment conversation. One of deployment_conversation_id or external_session_id must be provided.

        Returns:
            DeploymentConversationExport: The deployment conversation html export.
        """
        return self.client.export_deployment_conversation(self.deployment_conversation_id, external_session_id)
