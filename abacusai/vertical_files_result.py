from .return_class import AbstractApiClass


class VerticalFilesResult(AbstractApiClass):
    """
        Paginated result of vertical files listing.

        Args:
            client (ApiClient): An authenticated API Client instance
            files (list): List of file objects with document_upload_id, file_name, file_metadata, etc.
            maxCount (int): The total number of files.
    """

    def __init__(self, client, files=None, maxCount=None):
        super().__init__(client, None)
        self.files = files
        self.max_count = maxCount
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'files': repr(self.files),
                     f'max_count': repr(self.max_count)}
        class_name = "VerticalFilesResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'files': self.files, 'max_count': self.max_count}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
