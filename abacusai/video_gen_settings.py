from .return_class import AbstractApiClass


class VideoGenSettings(AbstractApiClass):
    """
        Video generation settings

        Args:
            client (ApiClient): An authenticated API Client instance
            prompt (dict): The prompt for the video.
            negativePrompt (dict): The negative prompt for the video.
            cfgScale (dict): The flexibility scale for video generation.
            mode (dict): The video generation mode (standard or professional).
            aspectRatio (dict): The aspect ratio of the video in seconds.
            duration (dict): The duration of the video.
            loop (dict): Whether the video should loop.
            startFrame (dict): The start frame of the video.
            endFrame (dict): The end frame of the video.
            rewritePrompt (dict): Whether to rewrite the prompt.
    """

    def __init__(self, client, prompt=None, negativePrompt=None, cfgScale=None, mode=None, aspectRatio=None, duration=None, loop=None, startFrame=None, endFrame=None, rewritePrompt=None):
        super().__init__(client, None)
        self.prompt = prompt
        self.negative_prompt = negativePrompt
        self.cfg_scale = cfgScale
        self.mode = mode
        self.aspect_ratio = aspectRatio
        self.duration = duration
        self.loop = loop
        self.start_frame = startFrame
        self.end_frame = endFrame
        self.rewrite_prompt = rewritePrompt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'prompt': repr(self.prompt), f'negative_prompt': repr(self.negative_prompt), f'cfg_scale': repr(self.cfg_scale), f'mode': repr(self.mode), f'aspect_ratio': repr(
            self.aspect_ratio), f'duration': repr(self.duration), f'loop': repr(self.loop), f'start_frame': repr(self.start_frame), f'end_frame': repr(self.end_frame), f'rewrite_prompt': repr(self.rewrite_prompt)}
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
        resp = {'prompt': self.prompt, 'negative_prompt': self.negative_prompt, 'cfg_scale': self.cfg_scale, 'mode': self.mode, 'aspect_ratio': self.aspect_ratio,
                'duration': self.duration, 'loop': self.loop, 'start_frame': self.start_frame, 'end_frame': self.end_frame, 'rewrite_prompt': self.rewrite_prompt}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
