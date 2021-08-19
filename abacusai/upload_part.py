

class UploadPart():
    '''
        Unique identifiers for a part
    '''

    def __init__(self, client, etag=None, md5=None):
        self.client = client
        self.id = None
        self.etag = etag
        self.md5 = md5

    def __repr__(self):
        return f"UploadPart(etag={repr(self.etag)}, md5={repr(self.md5)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'etag': self.etag, 'md5': self.md5}
