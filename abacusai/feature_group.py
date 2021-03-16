from .feature_column import FeatureColumn


class FeatureGroup():
    '''

    '''

    def __init__(self, client, featureGroupId=None, name=None, tableName=None, sql=None, datasetId=None, sourceTables=None, createdAt=None, description=None, datasetType=None, useForTraining=None, columns={}):
        self.client = client
        self.id = featureGroupId
        self.feature_group_id = featureGroupId
        self.name = name
        self.table_name = tableName
        self.sql = sql
        self.dataset_id = datasetId
        self.source_tables = sourceTables
        self.created_at = createdAt
        self.description = description
        self.dataset_type = datasetType
        self.use_for_training = useForTraining
        self.columns = client._build_class(FeatureColumn, columns)

    def __repr__(self):
        return f"FeatureGroup(feature_group_id={repr(self.feature_group_id)}, name={repr(self.name)}, table_name={repr(self.table_name)}, sql={repr(self.sql)}, dataset_id={repr(self.dataset_id)}, source_tables={repr(self.source_tables)}, created_at={repr(self.created_at)}, description={repr(self.description)}, dataset_type={repr(self.dataset_type)}, use_for_training={repr(self.use_for_training)}, columns={repr(self.columns)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_id': self.feature_group_id, 'name': self.name, 'table_name': self.table_name, 'sql': self.sql, 'dataset_id': self.dataset_id, 'source_tables': self.source_tables, 'created_at': self.created_at, 'description': self.description, 'dataset_type': self.dataset_type, 'use_for_training': self.use_for_training, 'columns': [elem.to_dict() for elem in self.columns or []]}
