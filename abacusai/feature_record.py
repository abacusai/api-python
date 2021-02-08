

class FeatureRecord():
    '''

    '''

    def __init__(self, client, data=None):
        self.client = client
        self.id = None
        self.data = data

    def __repr__(self):
        return f"FeatureRecord(data={repr(self.data)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'data': self.data}
