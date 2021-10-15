from .nested_feature import NestedFeature
from .point_in_time_feature import PointInTimeFeature


class Feature():
    '''
        A feature in a feature group
    '''

    def __init__(self, client, name=None, selectClause=None, featureMapping=None, sourceTable=None, originalName=None, usingClause=None, orderClause=None, whereClause=None, featureType=None, dataType=None, columns={}, pointInTimeInfo={}):
        self.client = client
        self.id = None
        self.name = name
        self.select_clause = selectClause
        self.feature_mapping = featureMapping
        self.source_table = sourceTable
        self.original_name = originalName
        self.using_clause = usingClause
        self.order_clause = orderClause
        self.where_clause = whereClause
        self.feature_type = featureType
        self.data_type = dataType
        self.columns = client._build_class(NestedFeature, columns)
        self.point_in_time_info = client._build_class(
            PointInTimeFeature, pointInTimeInfo)

    def __repr__(self):
        return f"Feature(name={repr(self.name)}, select_clause={repr(self.select_clause)}, feature_mapping={repr(self.feature_mapping)}, source_table={repr(self.source_table)}, original_name={repr(self.original_name)}, using_clause={repr(self.using_clause)}, order_clause={repr(self.order_clause)}, where_clause={repr(self.where_clause)}, feature_type={repr(self.feature_type)}, data_type={repr(self.data_type)}, columns={repr(self.columns)}, point_in_time_info={repr(self.point_in_time_info)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'name': self.name, 'select_clause': self.select_clause, 'feature_mapping': self.feature_mapping, 'source_table': self.source_table, 'original_name': self.original_name, 'using_clause': self.using_clause, 'order_clause': self.order_clause, 'where_clause': self.where_clause, 'feature_type': self.feature_type, 'data_type': self.data_type, 'columns': [elem.to_dict() for elem in self.columns or []], 'point_in_time_info': [elem.to_dict() for elem in self.point_in_time_info or []]}
