from .return_class import AbstractApiClass


class VideoGenSettings(AbstractApiClass):
    """
        Video generation settings

        Args:
            client (ApiClient): An authenticated API Client instance
            prompt (dict): The prompt for the video.
            aspectRatio (dict): The aspect ratio of the video.
            loop (dict): Whether the video should loop.
            rewritePrompt (dict): weather to rewrite the prompt.
            startFrame (dict): The start frame of the video.
            endFrame (dict): The end frame of the video.
    """

    def __init__(self, client, prompt=None, aspectRatio=None, loop=None, rewritePrompt=None, startFrame=None, endFrame=None):
        super().__init__(client, None)
        self.prompt = prompt
        self.aspect_ratio = aspectRatio
        self.loop = loop
        self.rewrite_prompt = rewritePrompt
        self.start_frame = startFrame
        self.end_frame = endFrame
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'prompt': repr(self.prompt), f'aspect_ratio': repr(self.aspect_ratio), f'loop': repr(
            self.loop), f'rewrite_prompt': repr(self.rewrite_prompt), f'start_frame': repr(self.start_frame), f'end_frame': repr(self.end_frame)}
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
        resp = {'prompt': self.prompt, 'aspect_ratio': self.aspect_ratio, 'loop': self.loop,
                'rewrite_prompt': self.rewrite_prompt, 'start_frame': self.start_frame, 'end_frame': self.end_frame}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
