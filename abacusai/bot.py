from .return_class import AbstractApiClass


class Bot(AbstractApiClass):
    """
        A persistent chat bot (long-running agent the user messages like a contact)

        Args:
            client (ApiClient): An authenticated API Client instance
            botId (str): The id of the bot.
            ownerUserId (str): The user id of the agent's owner.
            isOwner (bool): Whether the caller owns the agent; False for a teammate's agent shared with the caller (via its conversation).
            canEdit (bool): Whether the caller may edit, delete and re-share the agent (the owner, a teammate shared with the editor role, or any teammate with conversation access to the agent's project).
            ownerEmail (str): The owner's email; shared agents in _listBots only.
            name (str): The name of the bot.
            description (str): The description of the bot.
            instructions (str): The user-provided instructions/persona for the bot.
            personality (str): The bot's self-maintained personality (tone, voice, quirks).
            memory (str): The bot's persistent memory notes.
            avatar (str): The avatar for the bot.
            status (str): The current activity status of the bot (IDLE, WORKING, WAITING_FOR_USER).
            lifecycle (str): The lifecycle of the bot (ACTIVE, ARCHIVED).
            deploymentConversationId (str): The bot's persistent conversation id.
            chatllmProjectId (str): The project the agent belongs to (its files, skills and instructions apply, and every teammate who can see the project sees it and every project editor may edit it); None for a personal agent outside a project.
            externalApplicationId (str): The external application id associated with the bot.
            lastMessagePreview (str): A preview of the last message in the bot's conversation.
            lastMessageAt (int): Unix timestamp of the last message in the bot's conversation.
            unreadCount (int): The number of unread bot messages.
            createdAt (int): Unix timestamp of when the bot was created.
            schedule (dict): The bot's heartbeat schedule (frequency/time/day_of_week/timezone), if any.
            tasks (list): The bot's named scheduled tasks (daemon_task_id/name/prompt/schedule; event tasks also carry webhook_url); _describeBot only.
            channels (dict): The bot's configured messaging channels keyed by channel name; secret values are masked.
            learnedPersonality (str): Personality the bot learned on its own from user feedback; _describeBot only.
            learnedInstructions (str): Rules the bot learned on its own from user feedback; _describeBot only.
            profileHistory (list): Changelog of the bot's self-edits to the learned fields (t/field/old/new, chronological; undo entries flagged); _describeBot only.
            selfReview (dict): Latest self-review state (n/last_t/score/note); _describeBot only.
            playbooks (list): Reusable procedures the bot saved (name/body/t/uses/wins/losses, chronological); _describeBot only.
            proactiveBudget (int): User-set daily cap on proactive messages from high-frequency fires (hourly heartbeat + minute tasks); None when uncapped; _describeBot only.
            chattyGreetings (bool): Whether the agent says a short personal hello when the owner opens its chat in the web UI (default True); _describeBot only.
            quietHours (dict): {'start': 'HH:MM', 'end': 'HH:MM'} window (agent timezone) in which high-frequency fires stay silent; None when the owner turned quiet hours off; _describeBot only.
            agentNetwork (dict): Agent-to-agent messaging state (discoverable/own_agents/contacts); _describeBot only.
            pushNotificationsEnabled (bool): Whether the agent may send mobile push notifications (the agent conversation's own switch); _describeBot only.
    """

    def __init__(self, client, botId=None, ownerUserId=None, isOwner=None, canEdit=None, ownerEmail=None, name=None, description=None, instructions=None, personality=None, memory=None, avatar=None, status=None, lifecycle=None, deploymentConversationId=None, chatllmProjectId=None, externalApplicationId=None, lastMessagePreview=None, lastMessageAt=None, unreadCount=None, createdAt=None, schedule=None, tasks=None, channels=None, learnedPersonality=None, learnedInstructions=None, profileHistory=None, selfReview=None, playbooks=None, proactiveBudget=None, chattyGreetings=None, quietHours=None, agentNetwork=None, pushNotificationsEnabled=None):
        super().__init__(client, botId)
        self.bot_id = botId
        self.owner_user_id = ownerUserId
        self.is_owner = isOwner
        self.can_edit = canEdit
        self.owner_email = ownerEmail
        self.name = name
        self.description = description
        self.instructions = instructions
        self.personality = personality
        self.memory = memory
        self.avatar = avatar
        self.status = status
        self.lifecycle = lifecycle
        self.deployment_conversation_id = deploymentConversationId
        self.chatllm_project_id = chatllmProjectId
        self.external_application_id = externalApplicationId
        self.last_message_preview = lastMessagePreview
        self.last_message_at = lastMessageAt
        self.unread_count = unreadCount
        self.created_at = createdAt
        self.schedule = schedule
        self.tasks = tasks
        self.channels = channels
        self.learned_personality = learnedPersonality
        self.learned_instructions = learnedInstructions
        self.profile_history = profileHistory
        self.self_review = selfReview
        self.playbooks = playbooks
        self.proactive_budget = proactiveBudget
        self.chatty_greetings = chattyGreetings
        self.quiet_hours = quietHours
        self.agent_network = agentNetwork
        self.push_notifications_enabled = pushNotificationsEnabled
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'bot_id': repr(self.bot_id), f'owner_user_id': repr(self.owner_user_id), f'is_owner': repr(self.is_owner), f'can_edit': repr(self.can_edit), f'owner_email': repr(self.owner_email), f'name': repr(self.name), f'description': repr(self.description), f'instructions': repr(self.instructions), f'personality': repr(self.personality), f'memory': repr(self.memory), f'avatar': repr(self.avatar), f'status': repr(self.status), f'lifecycle': repr(self.lifecycle), f'deployment_conversation_id': repr(self.deployment_conversation_id), f'chatllm_project_id': repr(self.chatllm_project_id), f'external_application_id': repr(self.external_application_id), f'last_message_preview': repr(self.last_message_preview), f'last_message_at': repr(
            self.last_message_at), f'unread_count': repr(self.unread_count), f'created_at': repr(self.created_at), f'schedule': repr(self.schedule), f'tasks': repr(self.tasks), f'channels': repr(self.channels), f'learned_personality': repr(self.learned_personality), f'learned_instructions': repr(self.learned_instructions), f'profile_history': repr(self.profile_history), f'self_review': repr(self.self_review), f'playbooks': repr(self.playbooks), f'proactive_budget': repr(self.proactive_budget), f'chatty_greetings': repr(self.chatty_greetings), f'quiet_hours': repr(self.quiet_hours), f'agent_network': repr(self.agent_network), f'push_notifications_enabled': repr(self.push_notifications_enabled)}
        class_name = "Bot"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'bot_id': self.bot_id, 'owner_user_id': self.owner_user_id, 'is_owner': self.is_owner, 'can_edit': self.can_edit, 'owner_email': self.owner_email, 'name': self.name, 'description': self.description, 'instructions': self.instructions, 'personality': self.personality, 'memory': self.memory, 'avatar': self.avatar, 'status': self.status, 'lifecycle': self.lifecycle, 'deployment_conversation_id': self.deployment_conversation_id, 'chatllm_project_id': self.chatllm_project_id, 'external_application_id': self.external_application_id, 'last_message_preview': self.last_message_preview,
                'last_message_at': self.last_message_at, 'unread_count': self.unread_count, 'created_at': self.created_at, 'schedule': self.schedule, 'tasks': self.tasks, 'channels': self.channels, 'learned_personality': self.learned_personality, 'learned_instructions': self.learned_instructions, 'profile_history': self.profile_history, 'self_review': self.self_review, 'playbooks': self.playbooks, 'proactive_budget': self.proactive_budget, 'chatty_greetings': self.chatty_greetings, 'quiet_hours': self.quiet_hours, 'agent_network': self.agent_network, 'push_notifications_enabled': self.push_notifications_enabled}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
