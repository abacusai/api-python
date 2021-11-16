from .return_class import AbstractApiClass


class UseCase(AbstractApiClass):
    """
        A Project Use Case
    """

    def __init__(self, client, useCase=None, prettyName=None, description=None):
        super().__init__(client, None)
        self.use_case = useCase
        self.pretty_name = prettyName
        self.description = description

    def __repr__(self):
        return f"UseCase(use_case={repr(self.use_case)},\n  pretty_name={repr(self.pretty_name)},\n  description={repr(self.description)})"

    def to_dict(self):
        return {'use_case': self.use_case, 'pretty_name': self.pretty_name, 'description': self.description}
