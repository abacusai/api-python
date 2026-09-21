from .return_class import AbstractApiClass


class WhatsappReferralInvite(AbstractApiClass):
    """
        WhatsApp invites the AbacusAI Bot desktop app recorded for the invite milestone

        Args:
            client (ApiClient): An authenticated API Client instance
            newInvites (int): Recipients not invited by this user before.
            invitesSent (int): The user's total invites on record, every channel.
            milestoneCreditsGranted (int): Credits granted by this call when it crossed the milestone, else 0.
    """

    def __init__(self, client, newInvites=None, invitesSent=None, milestoneCreditsGranted=None):
        super().__init__(client, None)
        self.new_invites = newInvites
        self.invites_sent = invitesSent
        self.milestone_credits_granted = milestoneCreditsGranted
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'new_invites': repr(self.new_invites), f'invites_sent': repr(
            self.invites_sent), f'milestone_credits_granted': repr(self.milestone_credits_granted)}
        class_name = "WhatsappReferralInvite"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'new_invites': self.new_invites, 'invites_sent': self.invites_sent,
                'milestone_credits_granted': self.milestone_credits_granted}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
