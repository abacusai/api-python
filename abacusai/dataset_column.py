

class DatasetColumn():
    '''
        A schema description for a column
    '''

    def __init__(self, client, name=None, columnDataType=None, originalName=None):
        self.client = client
        self.id = None
        self.name = name
        self.column_data_type = columnDataType
        self.original_name = originalName

    def __repr__(self):
        return f"DatasetColumn(name={repr(self.name)}, column_data_type={repr(self.column_data_type)}, original_name={repr(self.original_name)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'column_data_type': self.column_data_type, 'original_name': self.original_name}
