from .return_class import AbstractApiClass


class VideoAsset(AbstractApiClass):
    """
        Video Asset — reusable generation collateral (a product or app today; extensible by asset_type).

        Args:
            client (ApiClient): An authenticated API Client instance
            videoAssetId (id): The ID of the video asset
            assetType (str): The kind of collateral (product, app)
            name (str): The asset's display name
            sourceUrl (str): The page the asset was scraped from, if any
            imageUrls (list): Signed URLs of the stored images (hero first), generated at read time
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
    """

    def __init__(self, client, videoAssetId=None, assetType=None, name=None, sourceUrl=None, imageUrls=None, createdAt=None, updatedAt=None):
        super().__init__(client, videoAssetId)
        self.video_asset_id = videoAssetId
        self.asset_type = assetType
        self.name = name
        self.source_url = sourceUrl
        self.image_urls = imageUrls
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'video_asset_id': repr(self.video_asset_id), f'asset_type': repr(self.asset_type), f'name': repr(self.name), f'source_url': repr(
            self.source_url), f'image_urls': repr(self.image_urls), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
        class_name = "VideoAsset"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'video_asset_id': self.video_asset_id, 'asset_type': self.asset_type, 'name': self.name,
                'source_url': self.source_url, 'image_urls': self.image_urls, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
