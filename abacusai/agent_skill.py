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
            display (bool): Whether this skill should be displayed prominently in the dropdown.
            globalSkill (bool): Whether this skill is an organization-level global skill.
            owned (bool): Whether the skill's files live in the scope being listed, so the caller can edit them.
            createdAt (str): The timestamp when the skill was created.
            updatedAt (str): The timestamp when the skill was last updated.
            accessLevel (str): Sharing audience of an owned user skill: PRIVATE, USER_GROUPS (specific people), or PUBLIC (anyone in the organization).
            sharedSkill (bool): Whether this skill is shared to the viewer by another user (use-only).
            sharedByEmail (str): Email of the user who shared this skill, on skills shared to the viewer.
            shareCount (int): Number of people an owned skill is shared with.
            sharedStoreDir (str): Directory name of the viewer's own copy of a shared skill, present once they opt in.
            hashedAgentSkillId (str): Hashed skill id used by skill share links.
            hashedOwnerUserId (str): Hashed id of the user whose store holds this skill's files.
    """

    def __init__(self, client, agentSkillId=None, skillName=None, description=None, skillDirectoryName=None, chatllmProjectId=None, systemCreated=None, enabled=None, default=None, display=None, globalSkill=None, owned=None, createdAt=None, updatedAt=None, accessLevel=None, sharedSkill=None, sharedByEmail=None, shareCount=None, sharedStoreDir=None, hashedAgentSkillId=None, hashedOwnerUserId=None):
        super().__init__(client, agentSkillId)
        self.agent_skill_id = agentSkillId
        self.skill_name = skillName
        self.description = description
        self.skill_directory_name = skillDirectoryName
        self.chatllm_project_id = chatllmProjectId
        self.system_created = systemCreated
        self.enabled = enabled
        self.default = default
        self.display = display
        self.global_skill = globalSkill
        self.owned = owned
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.access_level = accessLevel
        self.shared_skill = sharedSkill
        self.shared_by_email = sharedByEmail
        self.share_count = shareCount
        self.shared_store_dir = sharedStoreDir
        self.hashed_agent_skill_id = hashedAgentSkillId
        self.hashed_owner_user_id = hashedOwnerUserId
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'agent_skill_id': repr(self.agent_skill_id), f'skill_name': repr(self.skill_name), f'description': repr(self.description), f'skill_directory_name': repr(self.skill_directory_name), f'chatllm_project_id': repr(self.chatllm_project_id), f'system_created': repr(self.system_created), f'enabled': repr(self.enabled), f'default': repr(self.default), f'display': repr(self.display), f'global_skill': repr(self.global_skill), f'owned': repr(
            self.owned), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at), f'access_level': repr(self.access_level), f'shared_skill': repr(self.shared_skill), f'shared_by_email': repr(self.shared_by_email), f'share_count': repr(self.share_count), f'shared_store_dir': repr(self.shared_store_dir), f'hashed_agent_skill_id': repr(self.hashed_agent_skill_id), f'hashed_owner_user_id': repr(self.hashed_owner_user_id)}
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
        resp = {'agent_skill_id': self.agent_skill_id, 'skill_name': self.skill_name, 'description': self.description, 'skill_directory_name': self.skill_directory_name, 'chatllm_project_id': self.chatllm_project_id, 'system_created': self.system_created, 'enabled': self.enabled, 'default': self.default, 'display': self.display, 'global_skill': self.global_skill,
                'owned': self.owned, 'created_at': self.created_at, 'updated_at': self.updated_at, 'access_level': self.access_level, 'shared_skill': self.shared_skill, 'shared_by_email': self.shared_by_email, 'share_count': self.share_count, 'shared_store_dir': self.shared_store_dir, 'hashed_agent_skill_id': self.hashed_agent_skill_id, 'hashed_owner_user_id': self.hashed_owner_user_id}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
