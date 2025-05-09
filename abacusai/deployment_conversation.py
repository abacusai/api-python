from .deployment_conversation_event import DeploymentConversationEvent
from .hosted_artifact import HostedArtifact
from .return_class import AbstractApiClass


class DeploymentConversation(AbstractApiClass):
    """
        A deployment conversation.

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentConversationId (str): The unique identifier of the deployment conversation.
            name (str): The name of the deployment conversation.
            deploymentId (str): The deployment id associated with the deployment conversation.
            ownerUserId (str): The user id of the owner of the deployment conversation.
            createdAt (str): The timestamp at which the deployment conversation was created.
            lastEventCreatedAt (str): The timestamp at which the most recent corresponding deployment conversation event was created at.
            hasHistory (bool): Whether the deployment conversation has any history.
            externalSessionId (str): The external session id associated with the deployment conversation.
            regenerateAttempt (int): The sequence number of regeneration. Not regenerated if 0.
            externalApplicationId (str): The external application id associated with the deployment conversation.
            unusedDocumentUploadIds (list[str]): The list of unused document upload ids associated with the deployment conversation.
            humanizeInstructions (dict): Instructions for humanizing the conversation.
            conversationWarning (str): Extra text associated with the deployment conversation (to show it at the bottom of chatbot).
            conversationType (str): The type of the conversation, which depicts the application it caters to.
            metadata (dict): Additional backend information about the conversation.
            llmDisplayName (str): The display name of the LLM model used to generate the most recent response. Only used for system-created bots.
            llmBotIcon (str): The icon location of the LLM model used to generate the most recent response. Only used for system-created bots.
            searchSuggestions (list): The list of search suggestions for the conversation.
            chatllmTaskId (str): The chatllm task id associated with the deployment conversation.
            conversationStatus (str): The status of the deployment conversation (used for deep agent conversations).
            computerStatus (str): The status of the computer associated with the deployment conversation (used for deep agent conversations).
            totalEvents (int): The total number of events in the deployment conversation.
            contestNames (list[str]): Names of contests that this deployment is a part of.
            history (DeploymentConversationEvent): The history of the deployment conversation.
            hostedArtifacts (HostedArtifact): Artifacts that have been deployed by this conversation.
    """

    def __init__(self, client, deploymentConversationId=None, name=None, deploymentId=None, ownerUserId=None, createdAt=None, lastEventCreatedAt=None, hasHistory=None, externalSessionId=None, regenerateAttempt=None, externalApplicationId=None, unusedDocumentUploadIds=None, humanizeInstructions=None, conversationWarning=None, conversationType=None, metadata=None, llmDisplayName=None, llmBotIcon=None, searchSuggestions=None, chatllmTaskId=None, conversationStatus=None, computerStatus=None, totalEvents=None, contestNames=None, history={}, hostedArtifacts={}):
        super().__init__(client, deploymentConversationId)
        self.deployment_conversation_id = deploymentConversationId
        self.name = name
        self.deployment_id = deploymentId
        self.owner_user_id = ownerUserId
        self.created_at = createdAt
        self.last_event_created_at = lastEventCreatedAt
        self.has_history = hasHistory
        self.external_session_id = externalSessionId
        self.regenerate_attempt = regenerateAttempt
        self.external_application_id = externalApplicationId
        self.unused_document_upload_ids = unusedDocumentUploadIds
        self.humanize_instructions = humanizeInstructions
        self.conversation_warning = conversationWarning
        self.conversation_type = conversationType
        self.metadata = metadata
        self.llm_display_name = llmDisplayName
        self.llm_bot_icon = llmBotIcon
        self.search_suggestions = searchSuggestions
        self.chatllm_task_id = chatllmTaskId
        self.conversation_status = conversationStatus
        self.computer_status = computerStatus
        self.total_events = totalEvents
        self.contest_names = contestNames
        self.history = client._build_class(
            DeploymentConversationEvent, history)
        self.hosted_artifacts = client._build_class(
            HostedArtifact, hostedArtifacts)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_conversation_id': repr(self.deployment_conversation_id), f'name': repr(self.name), f'deployment_id': repr(self.deployment_id), f'owner_user_id': repr(self.owner_user_id), f'created_at': repr(self.created_at), f'last_event_created_at': repr(self.last_event_created_at), f'has_history': repr(self.has_history), f'external_session_id': repr(self.external_session_id), f'regenerate_attempt': repr(self.regenerate_attempt), f'external_application_id': repr(self.external_application_id), f'unused_document_upload_ids': repr(self.unused_document_upload_ids), f'humanize_instructions': repr(
            self.humanize_instructions), f'conversation_warning': repr(self.conversation_warning), f'conversation_type': repr(self.conversation_type), f'metadata': repr(self.metadata), f'llm_display_name': repr(self.llm_display_name), f'llm_bot_icon': repr(self.llm_bot_icon), f'search_suggestions': repr(self.search_suggestions), f'chatllm_task_id': repr(self.chatllm_task_id), f'conversation_status': repr(self.conversation_status), f'computer_status': repr(self.computer_status), f'total_events': repr(self.total_events), f'contest_names': repr(self.contest_names), f'history': repr(self.history), f'hosted_artifacts': repr(self.hosted_artifacts)}
        class_name = "DeploymentConversation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_conversation_id': self.deployment_conversation_id, 'name': self.name, 'deployment_id': self.deployment_id, 'owner_user_id': self.owner_user_id, 'created_at': self.created_at, 'last_event_created_at': self.last_event_created_at, 'has_history': self.has_history, 'external_session_id': self.external_session_id, 'regenerate_attempt': self.regenerate_attempt, 'external_application_id': self.external_application_id, 'unused_document_upload_ids': self.unused_document_upload_ids, 'humanize_instructions': self.humanize_instructions,
                'conversation_warning': self.conversation_warning, 'conversation_type': self.conversation_type, 'metadata': self.metadata, 'llm_display_name': self.llm_display_name, 'llm_bot_icon': self.llm_bot_icon, 'search_suggestions': self.search_suggestions, 'chatllm_task_id': self.chatllm_task_id, 'conversation_status': self.conversation_status, 'computer_status': self.computer_status, 'total_events': self.total_events, 'contest_names': self.contest_names, 'history': self._get_attribute_as_dict(self.history), 'hosted_artifacts': self._get_attribute_as_dict(self.hosted_artifacts)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def get(self, external_session_id: str = None, deployment_id: str = None, filter_intermediate_conversation_events: bool = True, get_unused_document_uploads: bool = False, start: int = None, limit: int = None):
        """
        Gets a deployment conversation.

        Args:
            external_session_id (str): External session ID of the conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            filter_intermediate_conversation_events (bool): If true, intermediate conversation events will be filtered out. Default is true.
            get_unused_document_uploads (bool): If true, unused document uploads will be returned. Default is false.
            start (int): The start index of the conversation.
            limit (int): The limit of the conversation.

        Returns:
            DeploymentConversation: The deployment conversation.
        """
        return self.client.get_deployment_conversation(self.deployment_conversation_id, external_session_id, deployment_id, filter_intermediate_conversation_events, get_unused_document_uploads, start, limit)

    def delete(self, deployment_id: str = None):
        """
        Delete a Deployment Conversation.

        Args:
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
        """
        return self.client.delete_deployment_conversation(self.deployment_conversation_id, deployment_id)

    def clear(self, external_session_id: str = None, deployment_id: str = None, user_message_indices: list = None):
        """
        Clear the message history of a Deployment Conversation.

        Args:
            external_session_id (str): The external session id associated with the deployment conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
            user_message_indices (list): Optional list of user message indices to clear. The associated bot response will also be cleared. If not provided, all messages will be cleared.
        """
        return self.client.clear_deployment_conversation(self.deployment_conversation_id, external_session_id, deployment_id, user_message_indices)

    def set_feedback(self, message_index: int, is_useful: bool = None, is_not_useful: bool = None, feedback: str = None, feedback_type: str = None, deployment_id: str = None):
        """
        Sets a deployment conversation message as useful or not useful

        Args:
            message_index (int): The index of the deployment conversation message
            is_useful (bool): If the message is useful. If true, the message is useful. If false, clear the useful flag.
            is_not_useful (bool): If the message is not useful. If true, the message is not useful. If set to false, clear the useful flag.
            feedback (str): Optional feedback on why the message is useful or not useful
            feedback_type (str): Optional feedback type
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
        """
        return self.client.set_deployment_conversation_feedback(self.deployment_conversation_id, message_index, is_useful, is_not_useful, feedback, feedback_type, deployment_id)

    def rename(self, name: str, deployment_id: str = None):
        """
        Rename a Deployment Conversation.

        Args:
            name (str): The new name of the conversation.
            deployment_id (str): The deployment this conversation belongs to. This is required if not logged in.
        """
        return self.client.rename_deployment_conversation(self.deployment_conversation_id, name, deployment_id)

    def export(self, external_session_id: str = None):
        """
        Export a Deployment Conversation.

        Args:
            external_session_id (str): The external session id associated with the deployment conversation. One of deployment_conversation_id or external_session_id must be provided.

        Returns:
            DeploymentConversationExport: The deployment conversation html export.
        """
        return self.client.export_deployment_conversation(self.deployment_conversation_id, external_session_id)

    def construct_agent_conversation_messages_for_llm(self, external_session_id: str = None, include_document_contents: bool = True):
        """
        Returns conversation history in a format for LLM calls.

        Args:
            external_session_id (str): External session ID of the conversation.
            include_document_contents (bool): If true, include contents from uploaded documents in the generated messages.

        Returns:
            AgentConversation: Contains a list of AgentConversationMessage that represents the conversation.
        """
        return self.client.construct_agent_conversation_messages_for_llm(self.deployment_conversation_id, external_session_id, include_document_contents)
