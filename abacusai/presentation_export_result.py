from .return_class import AbstractApiClass


class PresentationExportResult(AbstractApiClass):
    """
        Export Presentation

        Args:
            client (ApiClient): An authenticated API Client instance
            filePath (str): The path to the exported presentation
            webViewLink (str): The web view link to the exported presentation (if applicable)
    """

    def __init__(self, client, filePath=None, webViewLink=None):
        super().__init__(client, None)
        self.file_path = filePath
        self.web_view_link = webViewLink
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'file_path': repr(
            self.file_path), f'web_view_link': repr(self.web_view_link)}
        class_name = "PresentationExportResult"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'file_path': self.file_path,
                'web_view_link': self.web_view_link}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
