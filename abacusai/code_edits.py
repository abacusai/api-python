from .return_class import AbstractApiClass


class CodeEdits(AbstractApiClass):
    """
        A code edit response from an LLM

        Args:
            client (ApiClient): An authenticated API Client instance
            codeEdits (list[codeedit]): The code changes to be applied.
            codeChanges (list): The code changes to be applied.
    """

    def __init__(self, client, codeEdits=None, codeChanges=None):
        super().__init__(client, None)
        self.code_edits = codeEdits
        self.code_changes = codeChanges
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'code_edits': repr(
            self.code_edits), f'code_changes': repr(self.code_changes)}
        class_name = "CodeEdits"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'code_edits': self.code_edits,
                'code_changes': self.code_changes}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
