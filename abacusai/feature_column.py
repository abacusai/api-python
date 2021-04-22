

class FeatureColumn():
    '''

    '''

    def __init__(self, client, name=None, selectClause=None, dataType=None, dataUse=None, sourceTable=None, originalName=None):
        self.client = client
        self.id = None
        self.name = name
        self.select_clause = selectClause
        self.data_type = dataType
        self.data_use = dataUse
        self.source_table = sourceTable
        self.original_name = originalName

    def __repr__(self):
        return f"FeatureColumn(name={repr(self.name)}, select_clause={repr(self.select_clause)}, data_type={repr(self.data_type)}, data_use={repr(self.data_use)}, source_table={repr(self.source_table)}, original_name={repr(self.original_name)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'select_clause': self.select_clause, 'data_type': self.data_type, 'data_use': self.data_use, 'source_table': self.source_table, 'original_name': self.original_name}
