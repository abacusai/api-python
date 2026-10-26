from .return_class import AbstractApiClass


class OrganizationIngressPolicy(AbstractApiClass):
    """
        The organization's ingress policy: named network groups and where they apply (workspace ingress IP allowlist).

        Args:
            client (ApiClient): An authenticated API Client instance
            available (bool): Whether the feature is enabled for this organization and its cluster.
            enabled (bool): Whether the policy is enforced.
            groups (list): The network groups: group_id, name, entries (cidr, label).
            defaultGroupId (str): The group pre-attached to every app in an IP mode unless the app opts out.
            apps (list): Private hosted apps with their ingress access mode and attached groups.
    """

    def __init__(self, client, available=None, enabled=None, groups=None, defaultGroupId=None, apps=None):
        super().__init__(client, None)
        self.available = available
        self.enabled = enabled
        self.groups = groups
        self.default_group_id = defaultGroupId
        self.apps = apps
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'available': repr(self.available), f'enabled': repr(self.enabled), f'groups': repr(
            self.groups), f'default_group_id': repr(self.default_group_id), f'apps': repr(self.apps)}
        class_name = "OrganizationIngressPolicy"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'available': self.available, 'enabled': self.enabled, 'groups': self.groups,
                'default_group_id': self.default_group_id, 'apps': self.apps}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
