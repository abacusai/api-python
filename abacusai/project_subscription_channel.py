from .return_class import AbstractApiClass


class ProjectSubscriptionChannel(AbstractApiClass):
    """
        One inbound event channel (GitHub / webhook) wired to a project's coordinator

        Args:
            client (ApiClient): An authenticated API Client instance
            channel (str): github or webhook.
            label (str): Display label (repo name or the label given on add).
            secret (str): Signing / shared secret to configure on the sending side.
            webhookUrl (str): Signed URL the sending side posts to.
    """

    def __init__(self, client, channel=None, label=None, secret=None, webhookUrl=None):
        super().__init__(client, None)
        self.channel = channel
        self.label = label
        self.secret = secret
        self.webhook_url = webhookUrl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'channel': repr(self.channel), f'label': repr(
            self.label), f'secret': repr(self.secret), f'webhook_url': repr(self.webhook_url)}
        class_name = "ProjectSubscriptionChannel"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'channel': self.channel, 'label': self.label,
                'secret': self.secret, 'webhook_url': self.webhook_url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
