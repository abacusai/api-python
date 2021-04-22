

class Schema():
    '''

    '''

    def __init__(self, client, name=None, columnMapping=None, columnDataType=None, custom=None, sql=None, selectExpression=None):
        self.client = client
        self.id = None
        self.name = name
        self.column_mapping = columnMapping
        self.column_data_type = columnDataType
        self.custom = custom
        self.sql = sql
        self.select_expression = selectExpression

    def __repr__(self):
        return f"Schema(name={repr(self.name)}, column_mapping={repr(self.column_mapping)}, column_data_type={repr(self.column_data_type)}, custom={repr(self.custom)}, sql={repr(self.sql)}, select_expression={repr(self.select_expression)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'column_mapping': self.column_mapping, 'column_data_type': self.column_data_type, 'custom': self.custom, 'sql': self.sql, 'select_expression': self.select_expression}
