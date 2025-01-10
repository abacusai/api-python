from .return_class import AbstractApiClass


class CodeBot(AbstractApiClass):
    """
        A bot option for CodeLLM

        Args:
            client (ApiClient): An authenticated API Client instance
            llmName (str): The name of the LLM.
            name (str): The name of the bot.
            imageUploadSupported (bool): Whether the LLM supports image upload.
    """

    def __init__(self, client, llmName=None, name=None, imageUploadSupported=None):
        super().__init__(client, None)
        self.llm_name = llmName
        self.name = name
        self.image_upload_supported = imageUploadSupported
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'llm_name': repr(self.llm_name), f'name': repr(
            self.name), f'image_upload_supported': repr(self.image_upload_supported)}
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
        resp = {'llm_name': self.llm_name, 'name': self.name,
                'image_upload_supported': self.image_upload_supported}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
