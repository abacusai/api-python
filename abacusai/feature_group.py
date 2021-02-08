from .feature import Feature


class FeatureGroup():
    '''

    '''

    def __init__(self, client, featureGroupId=None, name=None, columns=None, sql=None, sourceDatasetIds=None, createdAt=None, features={}):
        self.client = client
        self.id = featureGroupId
        self.feature_group_id = featureGroupId
        self.name = name
        self.columns = columns
        self.sql = sql
        self.source_dataset_ids = sourceDatasetIds
        self.created_at = createdAt
        self.features = client._build_class(Feature, features)

    def __repr__(self):
        return f"FeatureGroup(feature_group_id={repr(self.feature_group_id)}, name={repr(self.name)}, columns={repr(self.columns)}, sql={repr(self.sql)}, source_dataset_ids={repr(self.source_dataset_ids)}, created_at={repr(self.created_at)}, features={repr(self.features)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_id': self.feature_group_id, 'name': self.name, 'columns': self.columns, 'sql': self.sql, 'source_dataset_ids': self.source_dataset_ids, 'created_at': self.created_at, 'features': self.features.to_dict() if self.features else None}
