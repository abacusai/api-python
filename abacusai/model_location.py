

class ModelLocation():
    '''

    '''

    def __init__(self, client, location=None, artifactNames=None):
        self.client = client
        self.id = None
        self.location = location
        self.artifact_names = artifactNames

    def __repr__(self):
        return f"ModelLocation(location={repr(self.location)}, artifact_names={repr(self.artifact_names)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'location': self.location, 'artifact_names': self.artifact_names}
