

class ProjectValidation():
    '''
        A validation result for a project
    '''

    def __init__(self, client, valid=None, datasetErrors=None, columnHints=None):
        self.client = client
        self.id = None
        self.valid = valid
        self.dataset_errors = datasetErrors
        self.column_hints = columnHints

    def __repr__(self):
        return f"ProjectValidation(valid={repr(self.valid)}, dataset_errors={repr(self.dataset_errors)}, column_hints={repr(self.column_hints)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'valid': self.valid, 'dataset_errors': self.dataset_errors, 'column_hints': self.column_hints}
