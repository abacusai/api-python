from .return_class import AbstractApiClass


class RegenerateLlmExternalApplication(AbstractApiClass):
    """
        An external application that specifies an LLM user can regenerate with in RouteLLM.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The display name of the LLM (ie GPT-5)
            llmBotIcon (str): The bot icon of the LLM.
            llmName (str): The external name of the LLM (ie OPENAI_GPT5)
    """

    def __init__(self, client, name=None, llmBotIcon=None, llmName=None):
        super().__init__(client, None)
        self.name = name
        self.llm_bot_icon = llmBotIcon
        self.llm_name = llmName
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'llm_bot_icon': repr(
            self.llm_bot_icon), f'llm_name': repr(self.llm_name)}
        class_name = "RegenerateLlmExternalApplication"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'llm_bot_icon': self.llm_bot_icon,
                'llm_name': self.llm_name}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
