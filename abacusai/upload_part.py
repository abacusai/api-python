from .return_class import AbstractApiClass


class UploadPart(AbstractApiClass):
    """
        Unique identifiers for a part
    """

    def __init__(self, client, etag=None, md5=None):
        super().__init__(client, None)
        self.etag = etag
        self.md5 = md5

    def __repr__(self):
        return f"UploadPart(etag={repr(self.etag)},\n  md5={repr(self.md5)})"

    def to_dict(self):
        return {'etag': self.etag, 'md5': self.md5}
