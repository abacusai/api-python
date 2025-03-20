from .return_class import AbstractApiClass


class FsEntry(AbstractApiClass):
    """
        File system entry.

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name of the file/folder
            type (str): The type of entry (file/folder)
            path (str): The path of the entry
            size (int): The size of the entry in bytes
            modified (int): The last modified timestamp
            isFolderEmpty (bool): Whether the folder is empty (only for folders)
            children (list): List of child FSEntry objects (only for folders)
    """

    def __init__(self, client, name=None, type=None, path=None, size=None, modified=None, isFolderEmpty=None, children=None):
        super().__init__(client, None)
        self.name = name
        self.type = type
        self.path = path
        self.size = size
        self.modified = modified
        self.isFolderEmpty = isFolderEmpty
        self.children = children
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'type': repr(self.type), f'path': repr(self.path), f'size': repr(
            self.size), f'modified': repr(self.modified), f'isFolderEmpty': repr(self.isFolderEmpty), f'children': repr(self.children)}
        class_name = "FsEntry"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'type': self.type, 'path': self.path, 'size': self.size,
                'modified': self.modified, 'isFolderEmpty': self.isFolderEmpty, 'children': self.children}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
