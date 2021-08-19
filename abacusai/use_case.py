

class UseCase():
    '''
        A Project Use Case
    '''

    def __init__(self, client, useCase=None, prettyName=None, description=None):
        self.client = client
        self.id = None
        self.use_case = useCase
        self.pretty_name = prettyName
        self.description = description

    def __repr__(self):
        return f"UseCase(use_case={repr(self.use_case)}, pretty_name={repr(self.pretty_name)}, description={repr(self.description)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'use_case': self.use_case, 'pretty_name': self.pretty_name, 'description': self.description}
