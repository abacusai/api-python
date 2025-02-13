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
            isPremium (bool): Whether the LLM is a premium LLM.
    """

    def __init__(self, client, llmName=None, name=None, imageUploadSupported=None, codeAgentSupported=None, isPremium=None):
        super().__init__(client, None)
        self.llm_name = llmName
        self.name = name
        self.image_upload_supported = imageUploadSupported
        self.code_agent_supported = codeAgentSupported
        self.is_premium = isPremium
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'llm_name': repr(self.llm_name), f'name': repr(self.name), f'image_upload_supported': repr(
            self.image_upload_supported), f'code_agent_supported': repr(self.code_agent_supported), f'is_premium': repr(self.is_premium)}
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
        resp = {'llm_name': self.llm_name, 'name': self.name, 'image_upload_supported': self.image_upload_supported,
                'code_agent_supported': self.code_agent_supported, 'is_premium': self.is_premium}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
