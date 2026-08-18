from .return_class import AbstractApiClass


class ProductLinkPreview(AbstractApiClass):
    """
        Product Link Preview — what a product page yielded, read before any generation is paid for.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The product name read off the page
            assetType (str): The kind of collateral the link resolves to (product, app)
            imageUrls (list): The page's gallery candidates, to hand back to createVideoAsset
    """

    def __init__(self, client, name=None, assetType=None, imageUrls=None):
        super().__init__(client, None)
        self.name = name
        self.asset_type = assetType
        self.image_urls = imageUrls
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'asset_type': repr(
            self.asset_type), f'image_urls': repr(self.image_urls)}
        class_name = "ProductLinkPreview"
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
                'image_urls': self.image_urls}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
