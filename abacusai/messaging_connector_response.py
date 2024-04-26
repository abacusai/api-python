from .return_class import AbstractApiClass


class MessagingConnectorResponse(AbstractApiClass):
    """
        The response to view label data for Teams

        Args:
            client (ApiClient): An authenticated API Client instance
            welcomeMessage (str): on the first installation of the app the user will get this message
            defaultMessage (str): when user triggers hi, hello, help they will get this message
            disclaimer (str): given along with every bot response
            messagingBotName (str): the name you want to see at various places instead of Abacus.AI
            useDefaultLabel (bool): to use the default Abacus.AI label in case it is set to true
            initAckReq (bool): Set to true if the initial Acknowledgment for the query is required by the user
            defaultLabels (dict): Dictionary of default labels, if the user-specified labels aren't set
            enabledExternalLinks (list): list of external application which have external links applicable
    """

    def __init__(self, client, welcomeMessage=None, defaultMessage=None, disclaimer=None, messagingBotName=None, useDefaultLabel=None, initAckReq=None, defaultLabels=None, enabledExternalLinks=None):
        super().__init__(client, None)
        self.welcome_message = welcomeMessage
        self.default_message = defaultMessage
        self.disclaimer = disclaimer
        self.messaging_bot_name = messagingBotName
        self.use_default_label = useDefaultLabel
        self.init_ack_req = initAckReq
        self.default_labels = defaultLabels
        self.enabled_external_links = enabledExternalLinks
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'welcome_message': repr(self.welcome_message), f'default_message': repr(self.default_message), f'disclaimer': repr(self.disclaimer), f'messaging_bot_name': repr(self.messaging_bot_name), f'use_default_label': repr(
            self.use_default_label), f'init_ack_req': repr(self.init_ack_req), f'default_labels': repr(self.default_labels), f'enabled_external_links': repr(self.enabled_external_links)}
        class_name = "MessagingConnectorResponse"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'welcome_message': self.welcome_message, 'default_message': self.default_message, 'disclaimer': self.disclaimer, 'messaging_bot_name': self.messaging_bot_name,
                'use_default_label': self.use_default_label, 'init_ack_req': self.init_ack_req, 'default_labels': self.default_labels, 'enabled_external_links': self.enabled_external_links}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
