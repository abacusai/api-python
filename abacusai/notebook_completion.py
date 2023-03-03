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

    def __repr__(self):
        return f"NotebookCompletion(cell_type={repr(self.cell_type)},\n  content={repr(self.content)},\n  mode={repr(self.mode)},\n  index={repr(self.index)},\n  prompts={repr(self.prompts)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'cell_type': self.cell_type, 'content': self.content, 'mode': self.mode, 'index': self.index, 'prompts': self.prompts}
