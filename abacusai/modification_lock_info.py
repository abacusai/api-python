from .return_class import AbstractApiClass


class ModificationLockInfo(AbstractApiClass):
    """
        Information about a modification lock for a certain object
    """

    def __init__(self, client, modificationLock=None, userEmails=None, organizationGroups=None):
        super().__init__(client, None)
        self.modification_lock = modificationLock
        self.user_emails = userEmails
        self.organization_groups = organizationGroups

    def __repr__(self):
        return f"ModificationLockInfo(modification_lock={repr(self.modification_lock)},\n  user_emails={repr(self.user_emails)},\n  organization_groups={repr(self.organization_groups)})"

    def to_dict(self):
        return {'modification_lock': self.modification_lock, 'user_emails': self.user_emails, 'organization_groups': self.organization_groups}
