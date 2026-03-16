from .return_class import AbstractApiClass
from .video_gen_model import VideoGenModel


class VideoGenSettings(AbstractApiClass):
    """
        Video generation settings

        Args:
            client (ApiClient): An authenticated API Client instance
            videoType (dict): Dropdown for type of video (text_to_video, image_to_video, lip_sync).
            modelsByType (dict): Maps each video type to the list of applicable model keys.
            imageFieldsByModel (dict): Maps each model to the list of image input field names.
            settings (dict): The settings for each model.
            warnings (dict): The warnings for each model.
            descriptions (dict): The descriptions for each model.
            model (VideoGenModel): Dropdown for models available for video generation.
    """

    def __init__(self, client, videoType=None, modelsByType=None, imageFieldsByModel=None, settings=None, warnings=None, descriptions=None, model={}):
        super().__init__(client, None)
        self.video_type = videoType
        self.models_by_type = modelsByType
        self.image_fields_by_model = imageFieldsByModel
        self.settings = settings
        self.warnings = warnings
        self.descriptions = descriptions
        self.model = client._build_class(VideoGenModel, model)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'video_type': repr(self.video_type), f'models_by_type': repr(self.models_by_type), f'image_fields_by_model': repr(
            self.image_fields_by_model), f'settings': repr(self.settings), f'warnings': repr(self.warnings), f'descriptions': repr(self.descriptions), f'model': repr(self.model)}
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
        resp = {'video_type': self.video_type, 'models_by_type': self.models_by_type, 'image_fields_by_model': self.image_fields_by_model,
                'settings': self.settings, 'warnings': self.warnings, 'descriptions': self.descriptions, 'model': self._get_attribute_as_dict(self.model)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
