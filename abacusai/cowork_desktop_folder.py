from .return_class import AbstractApiClass


class CoworkDesktopFolder(AbstractApiClass):
    """
        CoWork Desktop Folder

        Args:
            client (ApiClient): An authenticated API Client instance
            id (str): Opaque workspace id chosen by the desktop (maps to a local path on the host)
            label (str): Human-readable folder label shown on mobile
    """

    def __init__(self, client, id=None, label=None):
        super().__init__(client, None)
        self.id = id
        self.label = label
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'id': repr(self.id), f'label': repr(self.label)}
        class_name = "CoworkDesktopFolder"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'id': self.id, 'label': self.label}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
