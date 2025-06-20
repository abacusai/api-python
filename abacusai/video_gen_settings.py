from .return_class import AbstractApiClass
from .video_gen_model import VideoGenModel


class VideoGenSettings(AbstractApiClass):
    """
        Video generation settings

        Args:
            client (ApiClient): An authenticated API Client instance
            settings (dict): The settings for each model.
            model (VideoGenModel): Dropdown for models available for video generation.
    """

    def __init__(self, client, settings=None, model={}):
        super().__init__(client, None)
        self.settings = settings
        self.model = client._build_class(VideoGenModel, model)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'settings': repr(
            self.settings), f'model': repr(self.model)}
        class_name = "VideoGenSettings"
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
