from .return_class import AbstractApiClass
from .studio_showcase_item import StudioShowcaseItem


class StudioShowcase(AbstractApiClass):
    """
        Studio Showcase — the Explore wall's three tabs, each in display order.

        Args:
            client (ApiClient): An authenticated API Client instance
            featured (StudioShowcaseItem): The Featured tab
            images (StudioShowcaseItem): The Images tab
            videos (StudioShowcaseItem): The Videos tab
    """

    def __init__(self, client, featured={}, images={}, videos={}):
        super().__init__(client, None)
        self.featured = client._build_class(StudioShowcaseItem, featured)
        self.images = client._build_class(StudioShowcaseItem, images)
        self.videos = client._build_class(StudioShowcaseItem, videos)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'featured': repr(self.featured), f'images': repr(
            self.images), f'videos': repr(self.videos)}
        class_name = "StudioShowcase"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'featured': self._get_attribute_as_dict(self.featured), 'images': self._get_attribute_as_dict(
            self.images), 'videos': self._get_attribute_as_dict(self.videos)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
