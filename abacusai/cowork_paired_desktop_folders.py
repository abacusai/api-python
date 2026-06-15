from .cowork_desktop_folder import CoworkDesktopFolder
from .return_class import AbstractApiClass


class CoworkPairedDesktopFolders(AbstractApiClass):
    """
        CoWork Paired Desktop Folders

        Args:
            client (ApiClient): An authenticated API Client instance
            folders (CoworkDesktopFolder): Current workspace folders published by the target desktop
    """

    def __init__(self, client, folders={}):
        super().__init__(client, None)
        self.folders = client._build_class(CoworkDesktopFolder, folders)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'folders': repr(self.folders)}
        class_name = "CoworkPairedDesktopFolders"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'folders': self._get_attribute_as_dict(self.folders)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
