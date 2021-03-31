

class DataFilter():
    '''

    '''

    def __init__(self, client, whereExpression=None, type=None, join=None):
        self.client = client
        self.id = None
        self.where_expression = whereExpression
        self.type = type
        self.join = join

    def __repr__(self):
        return f"DataFilter(where_expression={repr(self.where_expression)}, type={repr(self.type)}, join={repr(self.join)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'where_expression': self.where_expression, 'type': self.type, 'join': self.join}
