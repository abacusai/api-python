from .return_class import AbstractApiClass


class BotInfo(AbstractApiClass):
    """
        Information about an external application and LLM.

        Args:
            client (ApiClient): An authenticated API Client instance
            externalApplicationId (str): The external application ID.
            llmName (str): The name of the LLM model. Only used for system-created bots.
            llmDisplayName (str): The display name of the LLM model. Only used for system-created bots.
            llmBotIcon (str): The icon location of the LLM model. Only used for system-created bots.
    """

    def __init__(self, client, externalApplicationId=None, llmName=None, llmDisplayName=None, llmBotIcon=None):
        super().__init__(client, None)
        self.external_application_id = externalApplicationId
        self.llm_name = llmName
        self.llm_display_name = llmDisplayName
        self.llm_bot_icon = llmBotIcon
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'external_application_id': repr(self.external_application_id), f'llm_name': repr(
            self.llm_name), f'llm_display_name': repr(self.llm_display_name), f'llm_bot_icon': repr(self.llm_bot_icon)}
        class_name = "BotInfo"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'external_application_id': self.external_application_id, 'llm_name': self.llm_name,
                'llm_display_name': self.llm_display_name, 'llm_bot_icon': self.llm_bot_icon}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
