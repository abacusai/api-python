from .return_class import AbstractApiClass


class ProjectValidation(AbstractApiClass):
    """
        A validation result for a project
    """

    def __init__(self, client, valid=None, datasetErrors=None, columnHints=None):
        super().__init__(client, None)
        self.valid = valid
        self.dataset_errors = datasetErrors
        self.column_hints = columnHints

    def __repr__(self):
        return f"ProjectValidation(valid={repr(self.valid)},\n  dataset_errors={repr(self.dataset_errors)},\n  column_hints={repr(self.column_hints)})"

    def to_dict(self):
        return {'valid': self.valid, 'dataset_errors': self.dataset_errors, 'column_hints': self.column_hints}
