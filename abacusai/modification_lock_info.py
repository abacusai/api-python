

class ModificationLockInfo():
    '''
        Information about a modification lock for a certain object
    '''

    def __init__(self, client, modificationLock=None, userEmails=None, organizationGroups=None):
        self.client = client
        self.id = None
        self.modification_lock = modificationLock
        self.user_emails = userEmails
        self.organization_groups = organizationGroups

    def __repr__(self):
        return f"ModificationLockInfo(modification_lock={repr(self.modification_lock)}, user_emails={repr(self.user_emails)}, organization_groups={repr(self.organization_groups)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'modification_lock': self.modification_lock, 'user_emails': self.user_emails, 'organization_groups': self.organization_groups}
