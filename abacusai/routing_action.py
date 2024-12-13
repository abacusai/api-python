from .return_class import AbstractApiClass


class RoutingAction(AbstractApiClass):
    """
        Routing action

        Args:
            client (ApiClient): An authenticated API Client instance
            id (str): The id of the routing action.
            title (str): The title of the routing action.
            prompt (str): The prompt of the routing action.
            placeholder (str): The placeholder of the routing action.
            value (str): The value of the routing action.
            displayName (str): The display name of the routing action.
            isLarge (bool): UI placement
            isMedium (bool): UI placement
            additionalInfo (dict): Additional information for the routing action.
    """

    def __init__(self, client, id=None, title=None, prompt=None, placeholder=None, value=None, displayName=None, isLarge=None, isMedium=None, additionalInfo=None):
        super().__init__(client, None)
        self.id = id
        self.title = title
        self.prompt = prompt
        self.placeholder = placeholder
        self.value = value
        self.display_name = displayName
        self.is_large = isLarge
        self.is_medium = isMedium
        self.additional_info = additionalInfo
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'id': repr(self.id), f'title': repr(self.title), f'prompt': repr(self.prompt), f'placeholder': repr(self.placeholder), f'value': repr(
            self.value), f'display_name': repr(self.display_name), f'is_large': repr(self.is_large), f'is_medium': repr(self.is_medium), f'additional_info': repr(self.additional_info)}
        class_name = "RoutingAction"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'id': self.id, 'title': self.title, 'prompt': self.prompt, 'placeholder': self.placeholder, 'value': self.value,
                'display_name': self.display_name, 'is_large': self.is_large, 'is_medium': self.is_medium, 'additional_info': self.additional_info}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
