from .return_class import AbstractApiClass


class BotCreationBlock(AbstractApiClass):
    """
        Why creating a personal agent would be refused for the user right now

        Args:
            client (ApiClient): An authenticated API Client instance
            reason (str): credits (out of credits: add credits or upgrade), credits_upgrade (out of credits on a tier that must upgrade), agent_limit (the plan's active-agent limit is reached), or None when a create would go through.
            maxAgents (int): The active-agent limit of a capped plan.
    """

    def __init__(self, client, reason=None, maxAgents=None):
        super().__init__(client, None)
        self.reason = reason
        self.max_agents = maxAgents
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'reason': repr(self.reason),
                     f'max_agents': repr(self.max_agents)}
        class_name = "BotCreationBlock"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'reason': self.reason, 'max_agents': self.max_agents}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
