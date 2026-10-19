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
    """

    def __init__(self, client, chatllmProjectId=None, name=None, description=None, customInstructions=None, createdAt=None, updatedAt=None, createdByUserId=None, createdByEmail=None, accessLevel=None, verticalType=None, preferences=None, orchestratorDeploymentConversationId=None, projectMode=None, coordinatorBotId=None, coordinatorDeploymentConversationId=None, coordinatorActiveRuns=None, coordinatorLastRunAt=None):
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
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'chatllm_project_id': repr(self.chatllm_project_id), f'name': repr(self.name), f'description': repr(self.description), f'custom_instructions': repr(self.custom_instructions), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'created_by_user_id': repr(self.created_by_user_id), f'created_by_email': repr(self.created_by_email), f'access_level': repr(self.access_level), f'vertical_type': repr(
            self.vertical_type), f'preferences': repr(self.preferences), f'orchestrator_deployment_conversation_id': repr(self.orchestrator_deployment_conversation_id), f'project_mode': repr(self.project_mode), f'coordinator_bot_id': repr(self.coordinator_bot_id), f'coordinator_deployment_conversation_id': repr(self.coordinator_deployment_conversation_id), f'coordinator_active_runs': repr(self.coordinator_active_runs), f'coordinator_last_run_at': repr(self.coordinator_last_run_at)}
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
        resp = {'chatllm_project_id': self.chatllm_project_id, 'name': self.name, 'description': self.description, 'custom_instructions': self.custom_instructions, 'created_at': self.created_at, 'updated_at': self.updated_at, 'created_by_user_id': self.created_by_user_id, 'created_by_email': self.created_by_email, 'access_level': self.access_level, 'vertical_type': self.vertical_type,
                'preferences': self.preferences, 'orchestrator_deployment_conversation_id': self.orchestrator_deployment_conversation_id, 'project_mode': self.project_mode, 'coordinator_bot_id': self.coordinator_bot_id, 'coordinator_deployment_conversation_id': self.coordinator_deployment_conversation_id, 'coordinator_active_runs': self.coordinator_active_runs, 'coordinator_last_run_at': self.coordinator_last_run_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
