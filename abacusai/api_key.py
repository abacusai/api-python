

class ApiKey():
    '''
        An API Key to authenticate requests to the Abacus.AI API
    '''

    def __init__(self, client, apiKeyId=None, apiKey=None, tag=None, createdAt=None):
        self.client = client
        self.id = apiKeyId
        self.api_key_id = apiKeyId
        self.api_key = apiKey
        self.tag = tag
        self.created_at = createdAt

    def __repr__(self):
        return f"ApiKey(api_key_id={repr(self.api_key_id)}, api_key={repr(self.api_key)}, tag={repr(self.tag)}, created_at={repr(self.created_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'api_key_id': self.api_key_id, 'api_key': self.api_key, 'tag': self.tag, 'created_at': self.created_at}

    def delete(self):
        return self.client.delete_api_key(self.api_key_id)
