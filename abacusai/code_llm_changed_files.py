from .return_class import AbstractApiClass


class CodeLlmChangedFiles(AbstractApiClass):
    """
        Code changed files

        Args:
            client (ApiClient): An authenticated API Client instance
            addedFiles (list): A list of added file paths.
            updatedFiles (list): A list of updated file paths.
            deletedFiles (list): A list of deleted file paths.
    """

    def __init__(self, client, addedFiles=None, updatedFiles=None, deletedFiles=None):
        super().__init__(client, None)
        self.added_files = addedFiles
        self.updated_files = updatedFiles
        self.deleted_files = deletedFiles
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'added_files': repr(self.added_files), f'updated_files': repr(
            self.updated_files), f'deleted_files': repr(self.deleted_files)}
        class_name = "CodeLlmChangedFiles"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'added_files': self.added_files,
                'updated_files': self.updated_files, 'deleted_files': self.deleted_files}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
