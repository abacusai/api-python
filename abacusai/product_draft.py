from .return_class import AbstractApiClass


class ProductDraft(AbstractApiClass):
    """
        Product Draft — the unsaved result of scraping a product/app URL (becomes a VideoAsset on create).

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The scraped display name
            assetType (str): Detected kind (app for app-store URLs, else product)
            sourceUrl (str): The scraped page URL
            imageUrls (list): Hero image URLs found on the page (remote; downloaded to S3 on createVideoAsset)
    """

    def __init__(self, client, name=None, assetType=None, sourceUrl=None, imageUrls=None):
        super().__init__(client, None)
        self.name = name
        self.asset_type = assetType
        self.source_url = sourceUrl
        self.image_urls = imageUrls
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'asset_type': repr(
            self.asset_type), f'source_url': repr(self.source_url), f'image_urls': repr(self.image_urls)}
        class_name = "ProductDraft"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'asset_type': self.asset_type,
                'source_url': self.source_url, 'image_urls': self.image_urls}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
