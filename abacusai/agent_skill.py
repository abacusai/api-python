from .return_class import AbstractApiClass


class AgentSkill(AbstractApiClass):
    """
        A skill that can be attached to an agent.

        Args:
            client (ApiClient): An authenticated API Client instance
            agentSkillId (str): The unique identifier of the skill.
            skillName (str): The name of the skill.
            description (str): A description of what the skill does.
            skillDirectoryName (str): The directory name where skill files are stored.
            chatllmProjectId (str): The project ID this skill is associated with.
            systemCreated (bool): Whether this skill was created by the system.
            enabled (bool): Whether the skill is currently enabled.
            default (bool): Whether this skill is a default skill.
            createdAt (str): The timestamp when the skill was created.
            updatedAt (str): The timestamp when the skill was last updated.
    """

    def __init__(self, client, agentSkillId=None, skillName=None, description=None, skillDirectoryName=None, chatllmProjectId=None, systemCreated=None, enabled=None, default=None, createdAt=None, updatedAt=None):
        super().__init__(client, agentSkillId)
        self.agent_skill_id = agentSkillId
        self.skill_name = skillName
        self.description = description
        self.skill_directory_name = skillDirectoryName
        self.chatllm_project_id = chatllmProjectId
        self.system_created = systemCreated
        self.enabled = enabled
        self.default = default
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'agent_skill_id': repr(self.agent_skill_id), f'skill_name': repr(self.skill_name), f'description': repr(self.description), f'skill_directory_name': repr(self.skill_directory_name), f'chatllm_project_id': repr(
            self.chatllm_project_id), f'system_created': repr(self.system_created), f'enabled': repr(self.enabled), f'default': repr(self.default), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "AgentSkill"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'agent_skill_id': self.agent_skill_id, 'skill_name': self.skill_name, 'description': self.description, 'skill_directory_name': self.skill_directory_name,
                'chatllm_project_id': self.chatllm_project_id, 'system_created': self.system_created, 'enabled': self.enabled, 'default': self.default, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
