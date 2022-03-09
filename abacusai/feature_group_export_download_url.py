from .return_class import AbstractApiClass


class FeatureGroupExportDownloadUrl(AbstractApiClass):
    """
        A Feature Group Export Download Url, which is used to download the feature group version

        Args:
            client (ApiClient): An authenticated API Client instance
            downloadUrl (str): The url of the download location
            expiresAt (str): String representation of datetime when the url expires
    """

    def __init__(self, client, downloadUrl=None, expiresAt=None):
        super().__init__(client, None)
        self.download_url = downloadUrl
        self.expires_at = expiresAt

    def __repr__(self):
        return f"FeatureGroupExportDownloadUrl(download_url={repr(self.download_url)},\n  expires_at={repr(self.expires_at)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'download_url': self.download_url, 'expires_at': self.expires_at}
