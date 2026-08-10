from .return_class import AbstractApiClass


class DaemonTaskPermissions(AbstractApiClass):
    """
        Daemon Task Sharing Permissions

        Args:
            client (ApiClient): An authenticated API Client instance
            daemonTaskId (id): The ID of the daemon task.
            userPermissions (list): List of tuples containing (user_id, permission).
            userGroupPermissions (list): List of tuples containing (user_group_id, permission).
            shareWithOrg (bool): Whether the task is shared org-wide (view-only for everyone in the org).
    """

    def __init__(self, client, daemonTaskId=None, userPermissions=None, userGroupPermissions=None, shareWithOrg=None):
        super().__init__(client, None)
        self.daemon_task_id = daemonTaskId
        self.user_permissions = userPermissions
        self.user_group_permissions = userGroupPermissions
        self.share_with_org = shareWithOrg
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'daemon_task_id': repr(self.daemon_task_id), f'user_permissions': repr(
            self.user_permissions), f'user_group_permissions': repr(self.user_group_permissions), f'share_with_org': repr(self.share_with_org)}
        class_name = "DaemonTaskPermissions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'daemon_task_id': self.daemon_task_id, 'user_permissions': self.user_permissions,
                'user_group_permissions': self.user_group_permissions, 'share_with_org': self.share_with_org}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
