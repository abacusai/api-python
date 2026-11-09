from .return_class import AbstractApiClass


class ChatllmProject(AbstractApiClass):
    """
        ChatLLM Project

        Args:
            client (ApiClient): An authenticated API Client instance
            chatllmProjectId (id): The ID of the chatllm project.
            name (str): The name of the chatllm project.
            description (str): The description of the chatllm project.
            customInstructions (str): The custom instructions of the chatllm project.
            createdAt (str): The creation time of the chatllm project.
            updatedAt (str): The update time of the chatllm project.
            createdByUserId (int): The ID of the user who created the chatllm project.
            createdByEmail (str): The creator's email; only when the list call asks for it (include_creator_emails).
            accessLevel (str): The access level of the chatllm project.
            verticalType (str): The vertical type (HEALTH or null for regular projects).
            preferences (dict): The preferences JSON for vertical projects.
            orchestratorDeploymentConversationId (id): The ID of the orchestrator deployment conversation.
            projectMode (str): SHARED_WORKSPACE (default) or AGENT_COORDINATOR (one coordinator agent runs the project).
            coordinatorBotId (id): The coordinator agent of an AGENT_COORDINATOR project; None otherwise.
            coordinatorDeploymentConversationId (id): The coordinator agent's conversation; None otherwise.
            coordinatorActiveRuns (int): Delegated runs of the coordinator still open (starting or running); None for other projects.
            coordinatorLastRunAt (str): ISO-8601 UTC start time of the coordinator's most recent delegated run; None if it has none.
            listenerSessionCount (int): Listener sessions filed in the project that the caller can see; only when the list call asks for it (include_listener_stats).
            listenerLastSessionAt (str): When the newest of those sessions was last active, ISO-8601; None without one.
            listenerDocumentCount (int): Listener sessions whose summary and transcript are in the project's files.
            fileCount (int): Files in the project.
            listenerOwnSessionCount (int): The caller's own Listener sessions filed in the project that can go in its files (something was recorded).
            listenerOwnDocumentCount (int): Of those, the ones whose summary and transcript are in the project's files.
            listenerAutoFiles (bool): Whether the caller adds their own Listener sessions to the project's files when they end.
    """

    def __init__(self, client, chatllmProjectId=None, name=None, description=None, customInstructions=None, createdAt=None, updatedAt=None, createdByUserId=None, createdByEmail=None, accessLevel=None, verticalType=None, preferences=None, orchestratorDeploymentConversationId=None, projectMode=None, coordinatorBotId=None, coordinatorDeploymentConversationId=None, coordinatorActiveRuns=None, coordinatorLastRunAt=None, listenerSessionCount=None, listenerLastSessionAt=None, listenerDocumentCount=None, fileCount=None, listenerOwnSessionCount=None, listenerOwnDocumentCount=None, listenerAutoFiles=None):
        super().__init__(client, chatllmProjectId)
        self.chatllm_project_id = chatllmProjectId
        self.name = name
        self.description = description
        self.custom_instructions = customInstructions
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.created_by_user_id = createdByUserId
        self.created_by_email = createdByEmail
        self.access_level = accessLevel
        self.vertical_type = verticalType
        self.preferences = preferences
        self.orchestrator_deployment_conversation_id = orchestratorDeploymentConversationId
        self.project_mode = projectMode
        self.coordinator_bot_id = coordinatorBotId
        self.coordinator_deployment_conversation_id = coordinatorDeploymentConversationId
        self.coordinator_active_runs = coordinatorActiveRuns
        self.coordinator_last_run_at = coordinatorLastRunAt
        self.listener_session_count = listenerSessionCount
        self.listener_last_session_at = listenerLastSessionAt
        self.listener_document_count = listenerDocumentCount
        self.file_count = fileCount
        self.listener_own_session_count = listenerOwnSessionCount
        self.listener_own_document_count = listenerOwnDocumentCount
        self.listener_auto_files = listenerAutoFiles
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_project_id': repr(self.chatllm_project_id), f'name': repr(self.name), f'description': repr(self.description), f'custom_instructions': repr(self.custom_instructions), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'created_by_user_id': repr(self.created_by_user_id), f'created_by_email': repr(self.created_by_email), f'access_level': repr(self.access_level), f'vertical_type': repr(self.vertical_type), f'preferences': repr(self.preferences), f'orchestrator_deployment_conversation_id': repr(self.orchestrator_deployment_conversation_id), f'project_mode': repr(self.project_mode), f'coordinator_bot_id': repr(
            self.coordinator_bot_id), f'coordinator_deployment_conversation_id': repr(self.coordinator_deployment_conversation_id), f'coordinator_active_runs': repr(self.coordinator_active_runs), f'coordinator_last_run_at': repr(self.coordinator_last_run_at), f'listener_session_count': repr(self.listener_session_count), f'listener_last_session_at': repr(self.listener_last_session_at), f'listener_document_count': repr(self.listener_document_count), f'file_count': repr(self.file_count), f'listener_own_session_count': repr(self.listener_own_session_count), f'listener_own_document_count': repr(self.listener_own_document_count), f'listener_auto_files': repr(self.listener_auto_files)}
        class_name = "ChatllmProject"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'chatllm_project_id': self.chatllm_project_id, 'name': self.name, 'description': self.description, 'custom_instructions': self.custom_instructions, 'created_at': self.created_at, 'updated_at': self.updated_at, 'created_by_user_id': self.created_by_user_id, 'created_by_email': self.created_by_email, 'access_level': self.access_level, 'vertical_type': self.vertical_type, 'preferences': self.preferences, 'orchestrator_deployment_conversation_id': self.orchestrator_deployment_conversation_id, 'project_mode': self.project_mode, 'coordinator_bot_id': self.coordinator_bot_id,
                'coordinator_deployment_conversation_id': self.coordinator_deployment_conversation_id, 'coordinator_active_runs': self.coordinator_active_runs, 'coordinator_last_run_at': self.coordinator_last_run_at, 'listener_session_count': self.listener_session_count, 'listener_last_session_at': self.listener_last_session_at, 'listener_document_count': self.listener_document_count, 'file_count': self.file_count, 'listener_own_session_count': self.listener_own_session_count, 'listener_own_document_count': self.listener_own_document_count, 'listener_auto_files': self.listener_auto_files}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
