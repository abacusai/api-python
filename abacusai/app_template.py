from .return_class import AbstractApiClass


class AppTemplate(AbstractApiClass):
    """
        App Template — a curated remixable web app in the Templates gallery (client-safe registry

        Args:
            client (ApiClient): An authenticated API Client instance
            templateId (str): Stable template identifier (registry key)
            templateType (str): The kind of template (web_app today)
            name (str): Card title
            description (str): Card one-liner
            longDescription (str): Detail-modal copy
            category (str): Gallery filter category
            tags (list): Search/filter tags
            techStack (list): Technologies shown in the detail modal
            thumbUrl (str): Card thumbnail URL
            videoUrl (str): Demo video URL for the detail modal
            templateUrl (str): Deployed demo app URL powering the live preview iframe
    """

    def __init__(self, client, templateId=None, templateType=None, name=None, description=None, longDescription=None, category=None, tags=None, techStack=None, thumbUrl=None, videoUrl=None, templateUrl=None):
        super().__init__(client, None)
        self.template_id = templateId
        self.template_type = templateType
        self.name = name
        self.description = description
        self.long_description = longDescription
        self.category = category
        self.tags = tags
        self.tech_stack = techStack
        self.thumb_url = thumbUrl
        self.video_url = videoUrl
        self.template_url = templateUrl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'template_id': repr(self.template_id), f'template_type': repr(self.template_type), f'name': repr(self.name), f'description': repr(self.description), f'long_description': repr(self.long_description), f'category': repr(
            self.category), f'tags': repr(self.tags), f'tech_stack': repr(self.tech_stack), f'thumb_url': repr(self.thumb_url), f'video_url': repr(self.video_url), f'template_url': repr(self.template_url)}
        class_name = "AppTemplate"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'template_id': self.template_id, 'template_type': self.template_type, 'name': self.name, 'description': self.description, 'long_description': self.long_description,
                'category': self.category, 'tags': self.tags, 'tech_stack': self.tech_stack, 'thumb_url': self.thumb_url, 'video_url': self.video_url, 'template_url': self.template_url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
