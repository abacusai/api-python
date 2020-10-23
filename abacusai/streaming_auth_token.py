

class StreamingAuthToken():
    '''

    '''

    def __init__(self, client, streamingToken=None, createdAt=None):
        self.client = client
        self.id = None
        self.streaming_token = streamingToken
        self.created_at = createdAt

    def __repr__(self):
        return f"StreamingAuthToken(streaming_token={repr(self.streaming_token)}, created_at={repr(self.created_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'streaming_token': self.streaming_token, 'created_at': self.created_at}
