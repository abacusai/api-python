from .return_class import AbstractApiClass


class ImageGenModelOptions(AbstractApiClass):
    """
        Image generation model options

        Args:
            client (ApiClient): An authenticated API Client instance
            keys (list): The keys of the image generation model options represented as the enum values.
            values (list): The display names of the image generation model options.
    """

    def __init__(self, client, keys=None, values=None):
        super().__init__(client, None)
        self.keys = keys
        self.values = values
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'keys': repr(self.keys), f'values': repr(self.values)}
        class_name = "ImageGenModelOptions"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'keys': self.keys, 'values': self.values}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
