from .return_class import AbstractApiClass


class BotChannelOptions(AbstractApiClass):
    """
        What the Personal Agents channel screens may offer on this cluster

        Args:
            client (ApiClient): An authenticated API Client instance
            sharedChannels (list): Channels this cluster runs a shared Abacus bot for, so they link by QR / deep link with no user-supplied tokens.
            telegramBotUsername (str): The shared Telegram bot's username, when there is one.
            channels (list): Every supported channel: channel, label, whether it is shared, whether the 'use your own bot' screen should offer it (offers_own_bot), and the credential form it takes (credential_fields: key/label/secret, in display order).
            slackWorkspace (dict): The admin-installed Slack workspace personal agents can be reached from (team_id, workspace_name, linked: whether this user tied their Slack account to Abacus.AI), when the org has one.
    """

    def __init__(self, client, sharedChannels=None, telegramBotUsername=None, channels=None, slackWorkspace=None):
        super().__init__(client, None)
        self.shared_channels = sharedChannels
        self.telegram_bot_username = telegramBotUsername
        self.channels = channels
        self.slack_workspace = slackWorkspace
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'shared_channels': repr(self.shared_channels), f'telegram_bot_username': repr(
            self.telegram_bot_username), f'channels': repr(self.channels), f'slack_workspace': repr(self.slack_workspace)}
        class_name = "BotChannelOptions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'shared_channels': self.shared_channels, 'telegram_bot_username': self.telegram_bot_username,
                'channels': self.channels, 'slack_workspace': self.slack_workspace}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
