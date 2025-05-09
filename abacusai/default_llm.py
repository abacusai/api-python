from .return_class import AbstractApiClass


class DefaultLlm(AbstractApiClass):
    """
        A default LLM.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the LLM.
            enum (str): The enum of the LLM.
    """

    def __init__(self, client, name=None, enum=None):
        super().__init__(client, None)
        self.name = name
        self.enum = enum
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'enum': repr(self.enum)}
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
        resp = {'name': self.name, 'enum': self.enum}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
