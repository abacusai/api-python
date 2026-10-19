from .return_class import AbstractApiClass


class AgentRail(AbstractApiClass):
    """
        The agent rail beside a personal agent's chat: what the agent knows about its owner

        Args:
            client (ApiClient): An authenticated API Client instance
            enabled (bool): Whether the chat has an agent rail (personal_agent_orchestration, the caller's own personal agent).
            memory (dict): sections (core/preferences/people/open_loops/threads -> items [{text, name?, detail?}] and updated_at epoch seconds) and lately (the newest daily note: date/text), from the agent's memory.
            brainTopics (list): The AI Brain topics the agent reads (name/description/updated_at); empty when the owner's AI Brain is off.
    """

    def __init__(self, client, enabled=None, memory=None, brainTopics=None):
        super().__init__(client, None)
        self.enabled = enabled
        self.memory = memory
        self.brain_topics = brainTopics
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'enabled': repr(self.enabled), f'memory': repr(
            self.memory), f'brain_topics': repr(self.brain_topics)}
        class_name = "AgentRail"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'enabled': self.enabled, 'memory': self.memory,
                'brain_topics': self.brain_topics}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
