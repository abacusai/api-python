from .feature_column import FeatureColumn
from .feature_group_version import FeatureGroupVersion


class FeatureGroup():
    '''
        A feature group
    '''

    def __init__(self, client, featureGroupId=None, name=None, tableName=None, sql=None, sourceTables=None, createdAt=None, description=None, featureGroupType=None, useForTraining=None, sqlError=None, latestVersionOutdated=None, columns={}, duplicateColumns={}, latestFeatureGroupVersion={}):
        self.client = client
        self.id = featureGroupId
        self.feature_group_id = featureGroupId
        self.name = name
        self.table_name = tableName
        self.sql = sql
        self.source_tables = sourceTables
        self.created_at = createdAt
        self.description = description
        self.feature_group_type = featureGroupType
        self.use_for_training = useForTraining
        self.sql_error = sqlError
        self.latest_version_outdated = latestVersionOutdated
        self.columns = client._build_class(FeatureColumn, columns)
        self.duplicate_columns = client._build_class(
            FeatureColumn, duplicateColumns)
        self.latest_feature_group_version = client._build_class(
            FeatureGroupVersion, latestFeatureGroupVersion)

    def __repr__(self):
        return f"FeatureGroup(feature_group_id={repr(self.feature_group_id)}, name={repr(self.name)}, table_name={repr(self.table_name)}, sql={repr(self.sql)}, source_tables={repr(self.source_tables)}, created_at={repr(self.created_at)}, description={repr(self.description)}, feature_group_type={repr(self.feature_group_type)}, use_for_training={repr(self.use_for_training)}, sql_error={repr(self.sql_error)}, latest_version_outdated={repr(self.latest_version_outdated)}, columns={repr(self.columns)}, duplicate_columns={repr(self.duplicate_columns)}, latest_feature_group_version={repr(self.latest_feature_group_version)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_id': self.feature_group_id, 'name': self.name, 'table_name': self.table_name, 'sql': self.sql, 'source_tables': self.source_tables, 'created_at': self.created_at, 'description': self.description, 'feature_group_type': self.feature_group_type, 'use_for_training': self.use_for_training, 'sql_error': self.sql_error, 'latest_version_outdated': self.latest_version_outdated, 'columns': [elem.to_dict() for elem in self.columns or []], 'duplicate_columns': [elem.to_dict() for elem in self.duplicate_columns or []], 'latest_feature_group_version': [elem.to_dict() for elem in self.latest_feature_group_version or []]}

    def get_schema(self, project_id=None):
        return self.client.get_feature_group_schema(self.feature_group_id, project_id)

    def set_column_data_type(self, column, data_type):
        return self.client.set_feature_group_column_data_type(self.feature_group_id, column, data_type)

    def add_feature(self, name, select_expression):
        return self.client.add_feature(self.feature_group_id, name, select_expression)

    def create_feature(self, name, select_expression):
        return self.client.create_feature(self.feature_group_id, name, select_expression)

    def add_nested_feature(self, nested_feature_name, table_name, using_clause, where_clause=None, order_clause=None):
        return self.client.add_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause)

    def create_nested_feature(self, nested_feature_name, table_name, using_clause, where_clause=None, order_clause=None):
        return self.client.create_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause)

    def update_nested_feature(self, nested_feature_name, table_name=None, using_clause=None, where_clause=None, order_clause=None, new_nested_feature_name=None):
        return self.client.update_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause, new_nested_feature_name)

    def delete_nested_feature(self, nested_feature_name):
        return self.client.delete_nested_feature(self.feature_group_id, nested_feature_name)

    def attach_to_project(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.attach_feature_group_to_project(self.feature_group_id, project_id, feature_group_type)

    def add_to_project(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.add_feature_group_to_project(self.feature_group_id, project_id, feature_group_type)

    def remove_from_project(self, project_id):
        return self.client.remove_feature_group_from_project(self.feature_group_id, project_id)

    def use_for_training(self, project_id, use_for_training=True):
        return self.client.use_feature_group_for_training(self.feature_group_id, project_id, use_for_training)

    def update_type(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.update_feature_group_type(self.feature_group_id, project_id, feature_group_type)

    def set_type(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.set_feature_group_type(self.feature_group_id, project_id, feature_group_type)

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_feature_group(self.feature_group_id)

    def update(self, sql=None, name=None, description=None):
        return self.client.update_feature_group(self.feature_group_id, sql, name, description)

    def update_function_definition(self, sql=None, name=None):
        return self.client.update_feature_group_function_definition(self.feature_group_id, sql, name)

    def update_feature(self, name, select_expression=None, new_name=None):
        return self.client.update_feature(self.feature_group_id, name, select_expression, new_name)

    def delete_feature(self, name):
        return self.client.delete_feature(self.feature_group_id, name)

    def delete(self):
        return self.client.delete_feature_group(self.feature_group_id)

    def create_version(self):
        return self.client.create_feature_group_version(self.feature_group_id)

    def list_versions(self, limit=100, start_after_version=None):
        return self.client.list_feature_group_versions(self.feature_group_id, limit, start_after_version)

    def wait_for_materialization(self, timeout=7200):
        return self.client._poll(self, {'PENDING', 'GENERATING'}, timeout=timeout)

    def get_status(self):
        return self.describe().latest_feature_group_version.status
