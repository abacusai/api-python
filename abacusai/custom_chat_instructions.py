from .return_class import AbstractApiClass


class CustomChatInstructions(AbstractApiClass):
    """
        Custom Chat Instructions

        Args:
            client (ApiClient): An authenticated API Client instance
            userInformationInstructions (str): The behavior instructions for the chat.
            responseInstructions (str): The response instructions for the chat.
            enableCodeExecution (bool): Whether or not code execution is enabled.
            enableImageGeneration (bool): Whether or not image generation is enabled.
            enableWebSearch (bool): Whether or not web search is enabled.
            enablePlayground (bool): Whether or not playground is enabled.
            experimentalFeatures (dict): Experimental features.
    """

    def __init__(self, client, userInformationInstructions=None, responseInstructions=None, enableCodeExecution=None, enableImageGeneration=None, enableWebSearch=None, enablePlayground=None, experimentalFeatures=None):
        super().__init__(client, None)
        self.user_information_instructions = userInformationInstructions
        self.response_instructions = responseInstructions
        self.enable_code_execution = enableCodeExecution
        self.enable_image_generation = enableImageGeneration
        self.enable_web_search = enableWebSearch
        self.enable_playground = enablePlayground
        self.experimental_features = experimentalFeatures
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'user_information_instructions': repr(self.user_information_instructions), f'response_instructions': repr(self.response_instructions), f'enable_code_execution': repr(self.enable_code_execution), f'enable_image_generation': repr(
            self.enable_image_generation), f'enable_web_search': repr(self.enable_web_search), f'enable_playground': repr(self.enable_playground), f'experimental_features': repr(self.experimental_features)}
        class_name = "CustomChatInstructions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'user_information_instructions': self.user_information_instructions, 'response_instructions': self.response_instructions, 'enable_code_execution': self.enable_code_execution,
                'enable_image_generation': self.enable_image_generation, 'enable_web_search': self.enable_web_search, 'enable_playground': self.enable_playground, 'experimental_features': self.experimental_features}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
