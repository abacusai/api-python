from .feature_column import FeatureColumn


class FeatureGroupVersion():
    '''
        A materialized version of a feature group
    '''

    def __init__(self, client, featureGroupVersion=None, sql=None, sourceTables=None, createdAt=None, status=None, error=None, columns={}):
        self.client = client
        self.id = featureGroupVersion
        self.feature_group_version = featureGroupVersion
        self.sql = sql
        self.source_tables = sourceTables
        self.created_at = createdAt
        self.status = status
        self.error = error
        self.columns = client._build_class(FeatureColumn, columns)

    def __repr__(self):
        return f"FeatureGroupVersion(feature_group_version={repr(self.feature_group_version)}, sql={repr(self.sql)}, source_tables={repr(self.source_tables)}, created_at={repr(self.created_at)}, status={repr(self.status)}, error={repr(self.error)}, columns={repr(self.columns)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_version': self.feature_group_version, 'sql': self.sql, 'source_tables': self.source_tables, 'created_at': self.created_at, 'status': self.status, 'error': self.error, 'columns': [elem.to_dict() for elem in self.columns or []]}

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_feature_group_version(self.feature_group_version, feature_group_version)

    def wait_for_results(self, timeout=3600):
        return self.client._poll(self, {'PENDING'}, timeout=timeout)

    def get_status(self):
        return self.client._call_api('describeFeatureGroupVersion', 'GET', query_params={'featureGroupVersion': self.feature_group_version}, parse_type=FeatureGroupVersion).status

    def describe(self):
        return self.client._call_api('describeFeatureGroupVersion', 'GET', query_params={'featureGroupVersion': self.feature_group_version}, parse_type=FeatureGroupVersion)
