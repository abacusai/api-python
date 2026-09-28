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
            mediaCapabilities (dict): Maps each model to its accepted media types and input slots (derived from settings).
            settings (dict): The settings for each model.
            warnings (dict): The warnings for each model.
            highCostModels (list): Models whose warning should render highlighted rather than muted.
            descriptions (dict): The descriptions for each model.
            audioModes (dict): Maps models without a generate_audio toggle to a fixed audio behaviour ('alwaysOn'/'none').
            chineseModelsDisabled (bool): Whether the org disables Chinese models, which routes Auto through a different table.
            studioMaxMode (bool): Whether the session is the internal max-quality Studio, which routes Auto through the max-mode table.
            model (VideoGenModel): Dropdown for models available for video generation.
    """

    def __init__(self, client, videoType=None, modelsByType=None, imageFieldsByModel=None, mediaCapabilities=None, settings=None, warnings=None, highCostModels=None, descriptions=None, audioModes=None, chineseModelsDisabled=None, studioMaxMode=None, model={}):
        super().__init__(client, None)
        self.video_type = videoType
        self.models_by_type = modelsByType
        self.image_fields_by_model = imageFieldsByModel
        self.media_capabilities = mediaCapabilities
        self.settings = settings
        self.warnings = warnings
        self.high_cost_models = highCostModels
        self.descriptions = descriptions
        self.audio_modes = audioModes
        self.chinese_models_disabled = chineseModelsDisabled
        self.studio_max_mode = studioMaxMode
        self.model = client._build_class(VideoGenModel, model)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'video_type': repr(self.video_type), f'models_by_type': repr(self.models_by_type), f'image_fields_by_model': repr(self.image_fields_by_model), f'media_capabilities': repr(self.media_capabilities), f'settings': repr(self.settings), f'warnings': repr(
            self.warnings), f'high_cost_models': repr(self.high_cost_models), f'descriptions': repr(self.descriptions), f'audio_modes': repr(self.audio_modes), f'chinese_models_disabled': repr(self.chinese_models_disabled), f'studio_max_mode': repr(self.studio_max_mode), f'model': repr(self.model)}
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
        resp = {'video_type': self.video_type, 'models_by_type': self.models_by_type, 'image_fields_by_model': self.image_fields_by_model, 'media_capabilities': self.media_capabilities, 'settings': self.settings, 'warnings': self.warnings,
                'high_cost_models': self.high_cost_models, 'descriptions': self.descriptions, 'audio_modes': self.audio_modes, 'chinese_models_disabled': self.chinese_models_disabled, 'studio_max_mode': self.studio_max_mode, 'model': self._get_attribute_as_dict(self.model)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
