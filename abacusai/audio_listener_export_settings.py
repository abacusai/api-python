from .return_class import AbstractApiClass


class AudioListenerExportSettings(AbstractApiClass):
    """
        Where a user's Listener session transcripts are automatically exported.

        Args:
            client (ApiClient): An authenticated API Client instance
            isEnabled (bool): Whether transcripts are automatically exported when a session ends.
            service (str): The file connector transcripts are exported to (GOOGLEDRIVEUSER, ONEDRIVE, BOX or DROPBOX).
            folderId (str): The connector folder id transcripts are written to, if one was chosen.
            isOrgEnabled (bool): Whether the organization allows transcript export at all.
    """

    def __init__(self, client, isEnabled=None, service=None, folderId=None, isOrgEnabled=None):
        super().__init__(client, None)
        self.is_enabled = isEnabled
        self.service = service
        self.folder_id = folderId
        self.is_org_enabled = isOrgEnabled
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'is_enabled': repr(self.is_enabled), f'service': repr(
            self.service), f'folder_id': repr(self.folder_id), f'is_org_enabled': repr(self.is_org_enabled)}
        class_name = "AudioListenerExportSettings"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'is_enabled': self.is_enabled, 'service': self.service,
                'folder_id': self.folder_id, 'is_org_enabled': self.is_org_enabled}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
