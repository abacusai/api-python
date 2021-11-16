from .return_class import AbstractApiClass


class DataFilter(AbstractApiClass):
    """
        A sql logic statement for including and excluding data from training
    """

    def __init__(self, client, sql=None, type=None, whereExpression=None, join=None):
        super().__init__(client, None)
        self.sql = sql
        self.type = type
        self.where_expression = whereExpression
        self.join = join

    def __repr__(self):
        return f"DataFilter(sql={repr(self.sql)},\n  type={repr(self.type)},\n  where_expression={repr(self.where_expression)},\n  join={repr(self.join)})"

    def to_dict(self):
        return {'sql': self.sql, 'type': self.type, 'where_expression': self.where_expression, 'join': self.join}
