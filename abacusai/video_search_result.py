from .return_class import AbstractApiClass


class VideoSearchResult(AbstractApiClass):
    """
        A single video search result.

        Args:
            client (ApiClient): An authenticated API Client instance
            title (str): The title of the video.
            url (str): The URL of the video.
            thumbnailUrl (str): The URL of the thumbnail of the video.
            motionThumbnailUrl (str): The URL of the motion thumbnail of the video.
            embedUrl (str): The URL of the embed of the video.
    """

    def __init__(self, client, title=None, url=None, thumbnailUrl=None, motionThumbnailUrl=None, embedUrl=None):
        super().__init__(client, None)
        self.title = title
        self.url = url
        self.thumbnail_url = thumbnailUrl
        self.motion_thumbnail_url = motionThumbnailUrl
        self.embed_url = embedUrl
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'title': repr(self.title), f'url': repr(self.url), f'thumbnail_url': repr(
            self.thumbnail_url), f'motion_thumbnail_url': repr(self.motion_thumbnail_url), f'embed_url': repr(self.embed_url)}
        class_name = "VideoSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'title': self.title, 'url': self.url, 'thumbnail_url': self.thumbnail_url,
                'motion_thumbnail_url': self.motion_thumbnail_url, 'embed_url': self.embed_url}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
