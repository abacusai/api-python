from .return_class import AbstractApiClass


class DefaultLlm(AbstractApiClass):
    """
        A default LLM.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the LLM.
            enum (str): The enum of the LLM.
            isDeprecated (bool): Whether the model is deprecated (enterprise footprint entries) — rendered greyed in the enable/disable menu.
            inPicker (bool): For a greyed deprecated entry, whether it's promoted into the new-chat picker.
    """

    def __init__(self, client, name=None, enum=None, isDeprecated=None, inPicker=None):
        super().__init__(client, None)
        self.name = name
        self.enum = enum
        self.is_deprecated = isDeprecated
        self.in_picker = inPicker
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'enum': repr(self.enum), f'is_deprecated': repr(
            self.is_deprecated), f'in_picker': repr(self.in_picker)}
        class_name = "DefaultLlm"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'enum': self.enum,
                'is_deprecated': self.is_deprecated, 'in_picker': self.in_picker}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
