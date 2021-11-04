from .return_class import AbstractApiClass


class FeatureRecord(AbstractApiClass):
    """
        A feature record
    """

    def __init__(self, client, data=None):
        super().__init__(client, None)
        self.data = data

    def __repr__(self):
        return f"FeatureRecord(data={repr(self.data)})"

    def to_dict(self):
        return {'data': self.data}
