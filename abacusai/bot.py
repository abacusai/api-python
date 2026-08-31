from .return_class import AbstractApiClass


class Bot(AbstractApiClass):
    """
        A persistent chat bot (long-running agent the user messages like a contact)

        Args:
            client (ApiClient): An authenticated API Client instance
            botId (str): The id of the bot.
            name (str): The name of the bot.
            description (str): The description of the bot.
            instructions (str): The user-provided instructions/persona for the bot.
            personality (str): The bot's self-maintained personality (tone, voice, quirks).
            memory (str): The bot's persistent memory notes.
            avatar (str): The avatar for the bot.
            status (str): The current activity status of the bot (IDLE, WORKING, WAITING_FOR_USER).
            lifecycle (str): The lifecycle of the bot (ACTIVE, ARCHIVED).
            deploymentConversationId (str): The bot's persistent conversation id.
            externalApplicationId (str): The external application id associated with the bot.
            lastMessagePreview (str): A preview of the last message in the bot's conversation.
            lastMessageAt (int): Unix timestamp of the last message in the bot's conversation.
            unreadCount (int): The number of unread bot messages.
            createdAt (int): Unix timestamp of when the bot was created.
            schedule (dict): The bot's heartbeat schedule (frequency/time/day_of_week/timezone), if any.
            tasks (list): The bot's named scheduled tasks (daemon_task_id/name/prompt/schedule); _describeBot only.
            channels (dict): The bot's configured messaging channels keyed by channel name; secret values are masked.
    """

    def __init__(self, client, botId=None, name=None, description=None, instructions=None, personality=None, memory=None, avatar=None, status=None, lifecycle=None, deploymentConversationId=None, externalApplicationId=None, lastMessagePreview=None, lastMessageAt=None, unreadCount=None, createdAt=None, schedule=None, tasks=None, channels=None):
        super().__init__(client, botId)
        self.bot_id = botId
        self.name = name
        self.description = description
        self.instructions = instructions
        self.personality = personality
        self.memory = memory
        self.avatar = avatar
        self.status = status
        self.lifecycle = lifecycle
        self.deployment_conversation_id = deploymentConversationId
        self.external_application_id = externalApplicationId
        self.last_message_preview = lastMessagePreview
        self.last_message_at = lastMessageAt
        self.unread_count = unreadCount
        self.created_at = createdAt
        self.schedule = schedule
        self.tasks = tasks
        self.channels = channels
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'bot_id': repr(self.bot_id), f'name': repr(self.name), f'description': repr(self.description), f'instructions': repr(self.instructions), f'personality': repr(self.personality), f'memory': repr(self.memory), f'avatar': repr(self.avatar), f'status': repr(self.status), f'lifecycle': repr(self.lifecycle), f'deployment_conversation_id': repr(
            self.deployment_conversation_id), f'external_application_id': repr(self.external_application_id), f'last_message_preview': repr(self.last_message_preview), f'last_message_at': repr(self.last_message_at), f'unread_count': repr(self.unread_count), f'created_at': repr(self.created_at), f'schedule': repr(self.schedule), f'tasks': repr(self.tasks), f'channels': repr(self.channels)}
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
        resp = {'bot_id': self.bot_id, 'name': self.name, 'description': self.description, 'instructions': self.instructions, 'personality': self.personality, 'memory': self.memory, 'avatar': self.avatar, 'status': self.status, 'lifecycle': self.lifecycle, 'deployment_conversation_id': self.deployment_conversation_id,
                'external_application_id': self.external_application_id, 'last_message_preview': self.last_message_preview, 'last_message_at': self.last_message_at, 'unread_count': self.unread_count, 'created_at': self.created_at, 'schedule': self.schedule, 'tasks': self.tasks, 'channels': self.channels}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
