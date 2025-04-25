from .image_gen_model_options import ImageGenModelOptions
from .return_class import AbstractApiClass


class ImageGenModel(AbstractApiClass):
    """
        Image generation model

        Args:
            client (ApiClient): An authenticated API Client instance
            displayName (str): 
            type (str): 
            valueType (str): 
            optional (bool): 
            default (str): 
            helptext (str): 
            options (ImageGenModelOptions): 
    """

    def __init__(self, client, displayName=None, type=None, valueType=None, optional=None, default=None, helptext=None, options={}):
        super().__init__(client, None)
        self.display_name = displayName
        self.type = type
        self.value_type = valueType
        self.optional = optional
        self.default = default
        self.helptext = helptext
        self.options = client._build_class(ImageGenModelOptions, options)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'display_name': repr(self.display_name), f'type': repr(self.type), f'value_type': repr(self.value_type), f'optional': repr(
            self.optional), f'default': repr(self.default), f'helptext': repr(self.helptext), f'options': repr(self.options)}
        class_name = "ImageGenModel"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'display_name': self.display_name, 'type': self.type, 'value_type': self.value_type, 'optional': self.optional,
                'default': self.default, 'helptext': self.helptext, 'options': self._get_attribute_as_dict(self.options)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
