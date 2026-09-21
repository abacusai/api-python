from .return_class import AbstractApiClass


class AgentNetwork(AbstractApiClass):
    """
        Agent-to-agent messaging state of a personal agent

        Args:
            client (ApiClient): An authenticated API Client instance
            discoverable (bool): Whether teammates in the org can find this agent and request a contact.
            orgHasOtherUsers (bool): Whether the owner's org has other users (teammate controls are shown only then).
            ownAgents (list): The owner's other agents (bot_id/name/description); always reachable.
            contacts (list): Other people's agents (bot_id/name/owner_user_id/owner_name/status/t/org_id/external); status is PENDING_IN, PENDING_OUT or ACCEPTED; external agents live in another org (org_id set) and may be on another cluster.
    """

    def __init__(self, client, discoverable=None, orgHasOtherUsers=None, ownAgents=None, contacts=None):
        super().__init__(client, None)
        self.discoverable = discoverable
        self.org_has_other_users = orgHasOtherUsers
        self.own_agents = ownAgents
        self.contacts = contacts
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'discoverable': repr(self.discoverable), f'org_has_other_users': repr(
            self.org_has_other_users), f'own_agents': repr(self.own_agents), f'contacts': repr(self.contacts)}
        class_name = "AgentNetwork"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'discoverable': self.discoverable, 'org_has_other_users': self.org_has_other_users,
                'own_agents': self.own_agents, 'contacts': self.contacts}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
