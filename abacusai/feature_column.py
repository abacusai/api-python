from .nested_column import NestedColumn


class FeatureColumn():
    '''
        A feature in a feature group
    '''

    def __init__(self, client, name=None, selectClause=None, columnDataType=None, columnMapping=None, sourceTable=None, originalName=None, usingClause=None, orderClause=None, whereClause=None, columns={}):
        self.client = client
        self.id = None
        self.name = name
        self.select_clause = selectClause
        self.column_data_type = columnDataType
        self.column_mapping = columnMapping
        self.source_table = sourceTable
        self.original_name = originalName
        self.using_clause = usingClause
        self.order_clause = orderClause
        self.where_clause = whereClause
        self.columns = client._build_class(NestedColumn, columns)

    def __repr__(self):
        return f"FeatureColumn(name={repr(self.name)}, select_clause={repr(self.select_clause)}, column_data_type={repr(self.column_data_type)}, column_mapping={repr(self.column_mapping)}, source_table={repr(self.source_table)}, original_name={repr(self.original_name)}, using_clause={repr(self.using_clause)}, order_clause={repr(self.order_clause)}, where_clause={repr(self.where_clause)}, columns={repr(self.columns)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'select_clause': self.select_clause, 'column_data_type': self.column_data_type, 'column_mapping': self.column_mapping, 'source_table': self.source_table, 'original_name': self.original_name, 'using_clause': self.using_clause, 'order_clause': self.order_clause, 'where_clause': self.where_clause, 'columns': [elem.to_dict() for elem in self.columns or []]}
