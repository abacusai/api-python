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

    def __repr__(self):
        return f"ProjectValidation(valid={repr(self.valid)},\n  dataset_errors={repr(self.dataset_errors)},\n  column_hints={repr(self.column_hints)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'valid': self.valid, 'dataset_errors': self.dataset_errors, 'column_hints': self.column_hints}
