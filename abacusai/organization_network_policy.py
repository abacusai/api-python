from .return_class import AbstractApiClass


class OrganizationNetworkPolicy(AbstractApiClass):
    """
        The organization's egress network policy (Agent Firewall) for agent computers.

        Args:
            client (ApiClient): An authenticated API Client instance
            enabled (bool): Whether the egress denylist is currently enforced for the organization.
            lifecycle (str): The policy lifecycle state (ACTIVE when enforced, DISABLED when off).
            egressFqdns (list): The hostnames blocked for the organization's agent computers.
            updatedAt (str): The timestamp at which the policy was last modified.
            available (bool): Whether the Agent Firewall is available for this organization.
    """

    def __init__(self, client, enabled=None, lifecycle=None, egressFqdns=None, updatedAt=None, available=None):
        super().__init__(client, None)
        self.enabled = enabled
        self.lifecycle = lifecycle
        self.egress_fqdns = egressFqdns
        self.updated_at = updatedAt
        self.available = available
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'enabled': repr(self.enabled), f'lifecycle': repr(self.lifecycle), f'egress_fqdns': repr(
            self.egress_fqdns), f'updated_at': repr(self.updated_at), f'available': repr(self.available)}
        class_name = "OrganizationNetworkPolicy"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'enabled': self.enabled, 'lifecycle': self.lifecycle,
                'egress_fqdns': self.egress_fqdns, 'updated_at': self.updated_at, 'available': self.available}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
