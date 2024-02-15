from .return_class import AbstractApiClass


class ProjectValidation(AbstractApiClass):
    """
        A validation result for a project

        Args:
            client (ApiClient): An authenticated API Client instance
            valid (bool): `true` if the project is valid and ready to be trained, otherwise `false`.
            datasetErrors (list[dict]): A list of errors keeping the dataset from being valid
            columnHints (dict): Hints for what to set on the columns
    """

    def __init__(self, client, valid=None, datasetErrors=None, columnHints=None):
        super().__init__(client, None)
        self.valid = valid
        self.dataset_errors = datasetErrors
        self.column_hints = columnHints
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'valid': repr(self.valid), f'dataset_errors': repr(
            self.dataset_errors), f'column_hints': repr(self.column_hints)}
        class_name = "ProjectValidation"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'valid': self.valid, 'dataset_errors': self.dataset_errors,
                'column_hints': self.column_hints}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}
