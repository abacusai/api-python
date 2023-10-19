from .code_source import CodeSource
from .return_class import AbstractApiClass


class Module(AbstractApiClass):
    """
        Customer created python module

        Args:
            client (ApiClient): An authenticated API Client instance
            name (str): The name to identify the algorithm. Only uppercase letters, numbers, and underscores are allowed.
            createdAt (str): The date and time when the Python function was created, in ISO-8601 format.
            notebookId (str): The unique string identifier of the notebook used to create or edit the module.
            codeSource (CodeSource): Information about the source code of the Python function.
    """

    def __init__(self, client, name=None, createdAt=None, notebookId=None, codeSource={}):
        super().__init__(client, None)
        self.name = name
        self.created_at = createdAt
        self.notebook_id = notebookId
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        repr_dict = {f'name': repr(self.name), f'created_at': repr(self.created_at), f'notebook_id': repr(
            self.notebook_id), f'code_source': repr(self.code_source)}
        class_name = "Module"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'name': self.name, 'created_at': self.created_at, 'notebook_id': self.notebook_id,
                'code_source': self._get_attribute_as_dict(self.code_source)}
        return {key: value for key, value in resp.items() if value is not None}
