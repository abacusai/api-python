from .return_class import AbstractApiClass


class SceneImageResult(AbstractApiClass):
    """
        Role Play scene-image generation result.

        Args:
            client (ApiClient): An authenticated API Client instance
            sceneImage (dict): The image just generated ({id, image_url, model, width, height, reference_image_url, created_at}). The image-gen prompt is intentionally omitted (kept internal, in debug info).
            sceneImages (list): All scene images stored on the message, oldest first (the full gallery).
    """

    def __init__(self, client, sceneImage=None, sceneImages=None):
        super().__init__(client, None)
        self.scene_image = sceneImage
        self.scene_images = sceneImages
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'scene_image': repr(
            self.scene_image), f'scene_images': repr(self.scene_images)}
        class_name = "SceneImageResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'scene_image': self.scene_image,
                'scene_images': self.scene_images}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
