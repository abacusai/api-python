from .return_class import AbstractApiClass


class FeatureGroupExportDownloadUrl(AbstractApiClass):
    """
        A Feature Group Export Download Url, which is used to download the feature group version

        Args:
            client (ApiClient): An authenticated API Client instance
            downloadUrl (str): The URL of the download location.
            expiresAt (str): String representation of the ISO-8601 datetime when the URL expires.
    """

    def __init__(self, client, downloadUrl=None, expiresAt=None):
        super().__init__(client, None)
        self.download_url = downloadUrl
        self.expires_at = expiresAt

    def __repr__(self):
        repr_dict = {f'download_url': repr(
            self.download_url), f'expires_at': repr(self.expires_at)}
        class_name = "FeatureGroupExportDownloadUrl"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'download_url': self.download_url,
                'expires_at': self.expires_at}
        return {key: value for key, value in resp.items() if value is not None}
