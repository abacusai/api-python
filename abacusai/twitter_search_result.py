from .return_class import AbstractApiClass


class TwitterSearchResult(AbstractApiClass):
    """
        A single twitter search result.

        Args:
            client (ApiClient): An authenticated API Client instance
            title (str): The title of the tweet.
            url (str): The URL of the tweet.
            twitterName (str): The name of the twitter user.
            twitterHandle (str): The handle of the twitter user.
            thumbnailUrl (str): The URL of the thumbnail of the tweet.
            thumbnailWidth (int): The width of the thumbnail of the tweet.
            thumbnailHeight (int): The height of the thumbnail of the tweet.
    """

    def __init__(self, client, title=None, url=None, twitterName=None, twitterHandle=None, thumbnailUrl=None, thumbnailWidth=None, thumbnailHeight=None):
        super().__init__(client, None)
        self.title = title
        self.url = url
        self.twitter_name = twitterName
        self.twitter_handle = twitterHandle
        self.thumbnail_url = thumbnailUrl
        self.thumbnail_width = thumbnailWidth
        self.thumbnail_height = thumbnailHeight
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'title': repr(self.title), f'url': repr(self.url), f'twitter_name': repr(self.twitter_name), f'twitter_handle': repr(
            self.twitter_handle), f'thumbnail_url': repr(self.thumbnail_url), f'thumbnail_width': repr(self.thumbnail_width), f'thumbnail_height': repr(self.thumbnail_height)}
        class_name = "TwitterSearchResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'title': self.title, 'url': self.url, 'twitter_name': self.twitter_name, 'twitter_handle': self.twitter_handle,
                'thumbnail_url': self.thumbnail_url, 'thumbnail_width': self.thumbnail_width, 'thumbnail_height': self.thumbnail_height}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
