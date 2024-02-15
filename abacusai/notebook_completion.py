from .return_class import AbstractApiClass


class NotebookCompletion(AbstractApiClass):
    """
        The result of a notebook code completion request

        Args:
            client (ApiClient): An authenticated API Client instance
            cellType (str): The type of the cell, either CODE or MARKDOWN
            content (str): The content of the cell
            mode (str): Either UPDATE or INSERT to dictate whether the completion will insert a new cell or update the last cell
            index (int): The index of the cell to insert after/ update
            prompts (list): The prompt(s) used to generate the completion
    """

    def __init__(self, client, cellType=None, content=None, mode=None, index=None, prompts=None):
        super().__init__(client, None)
        self.cell_type = cellType
        self.content = content
        self.mode = mode
        self.index = index
        self.prompts = prompts
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'cell_type': repr(self.cell_type), f'content': repr(self.content), f'mode': repr(
            self.mode), f'index': repr(self.index), f'prompts': repr(self.prompts)}
        class_name = "NotebookCompletion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'cell_type': self.cell_type, 'content': self.content,
                'mode': self.mode, 'index': self.index, 'prompts': self.prompts}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
