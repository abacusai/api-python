from .return_class import AbstractApiClass


class ModificationLockInfo(AbstractApiClass):
    """
        Information about a modification lock for a certain object

        Args:
            client (ApiClient): An authenticated API Client instance
            modificationLock (bool): Whether or not the object has its modification lock activated.
            userEmails (list of strings): The list of user emails allowed to modify the object if the object's modification lock is activated.
            organizationGroups (list of unique string identifiers): The list organization groups allowed to modify the object if the object's modification lock is activated.
    """

    def __init__(self, client, modificationLock=None, userEmails=None, organizationGroups=None):
        super().__init__(client, None)
        self.modification_lock = modificationLock
        self.user_emails = userEmails
        self.organization_groups = organizationGroups

    def __repr__(self):
        return f"ModificationLockInfo(modification_lock={repr(self.modification_lock)},\n  user_emails={repr(self.user_emails)},\n  organization_groups={repr(self.organization_groups)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'modification_lock': self.modification_lock, 'user_emails': self.user_emails, 'organization_groups': self.organization_groups}
