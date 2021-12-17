from .nested_feature import NestedFeature
from .point_in_time_feature import PointInTimeFeature
from .return_class import AbstractApiClass


class Feature(AbstractApiClass):
    """
        A feature in a feature group
    """

    def __init__(self, client, name=None, selectClause=None, featureMapping=None, sourceTable=None, originalName=None, usingClause=None, orderClause=None, whereClause=None, featureType=None, dataType=None, columns={}, pointInTimeInfo={}):
        super().__init__(client, None)
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
        return f"Feature(name={repr(self.name)},\n  select_clause={repr(self.select_clause)},\n  feature_mapping={repr(self.feature_mapping)},\n  source_table={repr(self.source_table)},\n  original_name={repr(self.original_name)},\n  using_clause={repr(self.using_clause)},\n  order_clause={repr(self.order_clause)},\n  where_clause={repr(self.where_clause)},\n  feature_type={repr(self.feature_type)},\n  data_type={repr(self.data_type)},\n  columns={repr(self.columns)},\n  point_in_time_info={repr(self.point_in_time_info)})"

    def to_dict(self):
        return {'name': self.name, 'select_clause': self.select_clause, 'feature_mapping': self.feature_mapping, 'source_table': self.source_table, 'original_name': self.original_name, 'using_clause': self.using_clause, 'order_clause': self.order_clause, 'where_clause': self.where_clause, 'feature_type': self.feature_type, 'data_type': self.data_type, 'columns': self._get_attribute_as_dict(self.columns), 'point_in_time_info': self._get_attribute_as_dict(self.point_in_time_info)}
