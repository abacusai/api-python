from .return_class import AbstractApiClass


class StudioShowcaseItem(AbstractApiClass):
    """
        Studio Showcase Item — one tile on Studio's Explore wall: a curated example generation, a Shorts format or an explainer template.

        Args:
            client (ApiClient): An authenticated API Client instance
            id (str): Stable tile identifier
            title (str): Display title
            modelName (str): Display name of the model credited for the example
            modelId (str): Gen-settings key of that model (`abacus_studio` when the example is attributed to Studio itself)
            type (str): image or video
            aspectRatio (str): CSS aspect ratio, e.g. "9 / 16"
            imageUrl (str): CDN URL of the still (the poster for a video tile)
            videoUrl (str): CDN URL of the full video, video tiles only
            videoThumbUrl (str): CDN URL of the compressed preview clip, video tiles only
            prompt (str): The prompt the example was generated with; empty for format and template tiles
            format (str): short (a Shorts format) or explainer (an explainer template); absent for a plain example
            adTemplateSlug (str): The Shorts format a short tile opens
    """

    def __init__(self, client, id=None, title=None, modelName=None, modelId=None, type=None, aspectRatio=None, imageUrl=None, videoUrl=None, videoThumbUrl=None, prompt=None, format=None, adTemplateSlug=None):
        super().__init__(client, None)
        self.id = id
        self.title = title
        self.model_name = modelName
        self.model_id = modelId
        self.type = type
        self.aspect_ratio = aspectRatio
        self.image_url = imageUrl
        self.video_url = videoUrl
        self.video_thumb_url = videoThumbUrl
        self.prompt = prompt
        self.format = format
        self.ad_template_slug = adTemplateSlug
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'id': repr(self.id), f'title': repr(self.title), f'model_name': repr(self.model_name), f'model_id': repr(self.model_id), f'type': repr(self.type), f'aspect_ratio': repr(self.aspect_ratio), f'image_url': repr(
            self.image_url), f'video_url': repr(self.video_url), f'video_thumb_url': repr(self.video_thumb_url), f'prompt': repr(self.prompt), f'format': repr(self.format), f'ad_template_slug': repr(self.ad_template_slug)}
        class_name = "StudioShowcaseItem"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'id': self.id, 'title': self.title, 'model_name': self.model_name, 'model_id': self.model_id, 'type': self.type, 'aspect_ratio': self.aspect_ratio, 'image_url': self.image_url,
                'video_url': self.video_url, 'video_thumb_url': self.video_thumb_url, 'prompt': self.prompt, 'format': self.format, 'ad_template_slug': self.ad_template_slug}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
