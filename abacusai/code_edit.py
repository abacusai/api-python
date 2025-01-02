from .return_class import AbstractApiClass


class CodeEdit(AbstractApiClass):
    """
        A code edit response from an LLM

        Args:
            client (ApiClient): An authenticated API Client instance
            filePath (str): The path of the file to be edited.
            startLine (int): The start line of the code to be replaced.
            endLine (int): The end line of the code to be replaced.
            text (str): The new text.
    """

    def __init__(self, client, filePath=None, startLine=None, endLine=None, text=None):
        super().__init__(client, None)
        self.file_path = filePath
        self.start_line = startLine
        self.end_line = endLine
        self.text = text
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'file_path': repr(self.file_path), f'start_line': repr(
            self.start_line), f'end_line': repr(self.end_line), f'text': repr(self.text)}
        class_name = "CodeEdit"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'file_path': self.file_path, 'start_line': self.start_line,
                'end_line': self.end_line, 'text': self.text}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
