from .return_class import AbstractApiClass


class DataFilter(AbstractApiClass):
    """
        A sql logic statement for including and excluding data from training

        Args:
            client (ApiClient): An authenticated API Client instance
            sql (str): [DEPRECATED] The sql logic for excluding data from this dataset
            type (str): Either INCLUDE or EXCLUDE
            whereExpression (str): The SQL WHERE expression for excluding or including data from this dataset
            join (str): The SQL operator to join with the following statement, if any
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
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'sql': self.sql, 'type': self.type, 'where_expression': self.where_expression, 'join': self.join}
