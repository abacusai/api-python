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
            sourceImageUrls (list): Signed URLs of the photos those images were generated from, if any
            urls (list): Extra product links attached as research material
            documentNames (list): Names of the uploaded documents attached as research material
            createdAt (str): The creation timestamp
            updatedAt (str): The last update timestamp
    """

    def __init__(self, client, videoAssetId=None, assetType=None, name=None, sourceUrl=None, imageUrls=None, sourceImageUrls=None, urls=None, documentNames=None, createdAt=None, updatedAt=None):
        super().__init__(client, videoAssetId)
        self.video_asset_id = videoAssetId
        self.asset_type = assetType
        self.name = name
        self.source_url = sourceUrl
        self.image_urls = imageUrls
        self.source_image_urls = sourceImageUrls
        self.urls = urls
        self.document_names = documentNames
        self.created_at = createdAt
        self.updated_at = updatedAt
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'video_asset_id': repr(self.video_asset_id), f'asset_type': repr(self.asset_type), f'name': repr(self.name), f'source_url': repr(self.source_url), f'image_urls': repr(
            self.image_urls), f'source_image_urls': repr(self.source_image_urls), f'urls': repr(self.urls), f'document_names': repr(self.document_names), f'created_at': repr(self.created_at), f'updated_at': repr(self.updated_at)}
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
        resp = {'video_asset_id': self.video_asset_id, 'asset_type': self.asset_type, 'name': self.name, 'source_url': self.source_url, 'image_urls': self.image_urls,
                'source_image_urls': self.source_image_urls, 'urls': self.urls, 'document_names': self.document_names, 'created_at': self.created_at, 'updated_at': self.updated_at}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
