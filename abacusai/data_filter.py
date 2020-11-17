

class DataFilter():
    '''

    '''

    def __init__(self, client, sql=None, type=None):
        self.client = client
        self.id = None
        self.sql = sql
        self.type = type

    def __repr__(self):
        return f"DataFilter(sql={repr(self.sql)}, type={repr(self.type)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'sql': self.sql, 'type': self.type}
