from .return_class import AbstractApiClass


class CodeBot(AbstractApiClass):
    """
        A bot option for CodeLLM

        Args:
            client (ApiClient): An authenticated API Client instance
            llmName (str): The name of the LLM.
            name (str): The name of the bot.
            imageUploadSupported (bool): Whether the LLM supports image upload.
            codeAgentSupported (bool): Whether the LLM supports code agent.
            codeEditSupported (bool): Whether the LLM supports code edit.
            isPremium (bool): Whether the LLM is a premium LLM.
            llmBotIcon (str): The icon of the LLM bot.
            provider (str): The provider of the LLM.
            isUserApiKeyAllowed (bool): Whether the LLM supports user API key.
            isRateLimited (bool): Whether the LLM is rate limited.
            apiKeyUrl (str): The URL to get the API key.
    """

    def __init__(self, client, llmName=None, name=None, imageUploadSupported=None, codeAgentSupported=None, codeEditSupported=None, isPremium=None, llmBotIcon=None, provider=None, isUserApiKeyAllowed=None, isRateLimited=None, apiKeyUrl=None):
        super().__init__(client, None)
        self.llm_name = llmName
        self.name = name
        self.image_upload_supported = imageUploadSupported
        self.code_agent_supported = codeAgentSupported
        self.code_edit_supported = codeEditSupported
        self.is_premium = isPremium
        self.llm_bot_icon = llmBotIcon
        self.provider = provider
        self.is_user_api_key_allowed = isUserApiKeyAllowed
        self.is_rate_limited = isRateLimited
        self.api_key_url = apiKeyUrl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'llm_name': repr(self.llm_name), f'name': repr(self.name), f'image_upload_supported': repr(self.image_upload_supported), f'code_agent_supported': repr(self.code_agent_supported), f'code_edit_supported': repr(self.code_edit_supported), f'is_premium': repr(
            self.is_premium), f'llm_bot_icon': repr(self.llm_bot_icon), f'provider': repr(self.provider), f'is_user_api_key_allowed': repr(self.is_user_api_key_allowed), f'is_rate_limited': repr(self.is_rate_limited), f'api_key_url': repr(self.api_key_url)}
        class_name = "CodeBot"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'llm_name': self.llm_name, 'name': self.name, 'image_upload_supported': self.image_upload_supported, 'code_agent_supported': self.code_agent_supported, 'code_edit_supported': self.code_edit_supported,
                'is_premium': self.is_premium, 'llm_bot_icon': self.llm_bot_icon, 'provider': self.provider, 'is_user_api_key_allowed': self.is_user_api_key_allowed, 'is_rate_limited': self.is_rate_limited, 'api_key_url': self.api_key_url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
