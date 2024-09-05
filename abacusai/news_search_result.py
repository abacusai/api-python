from .return_class import AbstractApiClass


class NewsSearchResult(AbstractApiClass):
    """
        A single news search result.

        Args:
            client (ApiClient): An authenticated API Client instance
            title (str): The title of the news.
            url (str): The URL of the news.
            description (str): The description of the news.
            thumbnailUrl (str): The URL of the image of the news.
            thumbnailWidth (int): The width of the image of the news.
            thumbnailHeight (int): The height of the image of the news.
            faviconUrl (str): The URL of the favicon of the news.
            datePublished (str): The date the news was published.
    """

    def __init__(self, client, title=None, url=None, description=None, thumbnailUrl=None, thumbnailWidth=None, thumbnailHeight=None, faviconUrl=None, datePublished=None):
        super().__init__(client, None)
        self.title = title
        self.url = url
        self.description = description
        self.thumbnail_url = thumbnailUrl
        self.thumbnail_width = thumbnailWidth
        self.thumbnail_height = thumbnailHeight
        self.favicon_url = faviconUrl
        self.date_published = datePublished
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'title': repr(self.title), f'url': repr(self.url), f'description': repr(self.description), f'thumbnail_url': repr(self.thumbnail_url), f'thumbnail_width': repr(
            self.thumbnail_width), f'thumbnail_height': repr(self.thumbnail_height), f'favicon_url': repr(self.favicon_url), f'date_published': repr(self.date_published)}
        class_name = "NewsSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'title': self.title, 'url': self.url, 'description': self.description, 'thumbnail_url': self.thumbnail_url, 'thumbnail_width':
                self.thumbnail_width, 'thumbnail_height': self.thumbnail_height, 'favicon_url': self.favicon_url, 'date_published': self.date_published}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
