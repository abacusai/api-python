from .image_gen_model import ImageGenModel
from .return_class import AbstractApiClass


class ImageGenSettings(AbstractApiClass):
    """
        Image generation settings

        Args:
            client (ApiClient): An authenticated API Client instance
            settings (dict): The settings for each model.
            model (ImageGenModel): Dropdown for models available for image generation.
    """

    def __init__(self, client, settings=None, model={}):
        super().__init__(client, None)
        self.settings = settings
        self.model = client._build_class(ImageGenModel, model)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'settings': repr(
            self.settings), f'model': repr(self.model)}
        class_name = "ImageGenSettings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'settings': self.settings,
                'model': self._get_attribute_as_dict(self.model)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
