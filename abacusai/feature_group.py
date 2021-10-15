from .feature import Feature
from .modification_lock_info import ModificationLockInfo
from .feature_group_version import FeatureGroupVersion


class FeatureGroup():
    '''
        A feature group
    '''

    def __init__(self, client, modificationLock=None, featureGroupId=None, name=None, featureGroupSourceType=None, tableName=None, sql=None, datasetId=None, functionSourceCode=None, functionName=None, sourceTables=None, createdAt=None, description=None, featureGroupType=None, useForTraining=None, sqlError=None, latestVersionOutdated=None, tags=None, primaryKey=None, updateTimestampKey=None, lookupKeys=None, features={}, duplicateFeatures={}, latestFeatureGroupVersion={}):
        self.client = client
        self.id = featureGroupId
        self.modification_lock = modificationLock
        self.feature_group_id = featureGroupId
        self.name = name
        self.feature_group_source_type = featureGroupSourceType
        self.table_name = tableName
        self.sql = sql
        self.dataset_id = datasetId
        self.function_source_code = functionSourceCode
        self.function_name = functionName
        self.source_tables = sourceTables
        self.created_at = createdAt
        self.description = description
        self.feature_group_type = featureGroupType
        self.use_for_training = useForTraining
        self.sql_error = sqlError
        self.latest_version_outdated = latestVersionOutdated
        self.tags = tags
        self.primary_key = primaryKey
        self.update_timestamp_key = updateTimestampKey
        self.lookup_keys = lookupKeys
        self.features = client._build_class(Feature, features)
        self.duplicate_features = client._build_class(
            Feature, duplicateFeatures)
        self.latest_feature_group_version = client._build_class(
            FeatureGroupVersion, latestFeatureGroupVersion)

    def __repr__(self):
        return f"FeatureGroup(modification_lock={repr(self.modification_lock)}, feature_group_id={repr(self.feature_group_id)}, name={repr(self.name)}, feature_group_source_type={repr(self.feature_group_source_type)}, table_name={repr(self.table_name)}, sql={repr(self.sql)}, dataset_id={repr(self.dataset_id)}, function_source_code={repr(self.function_source_code)}, function_name={repr(self.function_name)}, source_tables={repr(self.source_tables)}, created_at={repr(self.created_at)}, description={repr(self.description)}, feature_group_type={repr(self.feature_group_type)}, use_for_training={repr(self.use_for_training)}, sql_error={repr(self.sql_error)}, latest_version_outdated={repr(self.latest_version_outdated)}, tags={repr(self.tags)}, primary_key={repr(self.primary_key)}, update_timestamp_key={repr(self.update_timestamp_key)}, lookup_keys={repr(self.lookup_keys)}, features={repr(self.features)}, duplicate_features={repr(self.duplicate_features)}, latest_feature_group_version={repr(self.latest_feature_group_version)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'modification_lock': self.modification_lock, 'feature_group_id': self.feature_group_id, 'name': self.name, 'feature_group_source_type': self.feature_group_source_type, 'table_name': self.table_name, 'sql': self.sql, 'dataset_id': self.dataset_id, 'function_source_code': self.function_source_code, 'function_name': self.function_name, 'source_tables': self.source_tables, 'created_at': self.created_at, 'description': self.description, 'feature_group_type': self.feature_group_type, 'use_for_training': self.use_for_training, 'sql_error': self.sql_error, 'latest_version_outdated': self.latest_version_outdated, 'tags': self.tags, 'primary_key': self.primary_key, 'update_timestamp_key': self.update_timestamp_key, 'lookup_keys': self.lookup_keys, 'features': self.features.to_dict() if self.features else None, 'duplicate_features': [elem.to_dict() for elem in self.duplicate_features or []], 'latest_feature_group_version': [elem.to_dict() for elem in self.latest_feature_group_version or []]}

    def get_schema(self, project_id=None):
        return self.client.get_feature_group_schema(self.feature_group_id, project_id)

    def attach_to_project(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.attach_feature_group_to_project(self.feature_group_id, project_id, feature_group_type)

    def add_to_project(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.add_feature_group_to_project(self.feature_group_id, project_id, feature_group_type)

    def remove_from_project(self, project_id):
        return self.client.remove_feature_group_from_project(self.feature_group_id, project_id)

    def update_type(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.update_feature_group_type(self.feature_group_id, project_id, feature_group_type)

    def set_type(self, project_id, feature_group_type='CUSTOM_TABLE'):
        return self.client.set_feature_group_type(self.feature_group_id, project_id, feature_group_type)

    def use_for_training(self, project_id, use_for_training=True):
        return self.client.use_feature_group_for_training(self.feature_group_id, project_id, use_for_training)

    def set_schema(self, schema):
        return self.client.set_feature_group_schema(self.feature_group_id, schema)

    def add_feature(self, name, select_expression):
        return self.client.add_feature(self.feature_group_id, name, select_expression)

    def create_feature(self, name, select_expression):
        return self.client.create_feature(self.feature_group_id, name, select_expression)

    def add_tag(self, tag):
        return self.client.add_feature_group_tag(self.feature_group_id, tag)

    def remove_tag(self, tag):
        return self.client.remove_feature_group_tag(self.feature_group_id, tag)

    def add_nested_feature(self, nested_feature_name, table_name, using_clause, where_clause=None, order_clause=None):
        return self.client.add_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause)

    def create_nested_feature(self, nested_feature_name, table_name, using_clause, where_clause=None, order_clause=None):
        return self.client.create_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause)

    def update_nested_feature(self, nested_feature_name, table_name=None, using_clause=None, where_clause=None, order_clause=None, new_nested_feature_name=None):
        return self.client.update_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause, new_nested_feature_name)

    def delete_nested_feature(self, nested_feature_name):
        return self.client.delete_nested_feature(self.feature_group_id, nested_feature_name)

    def create_point_in_time_feature(self, feature_name, history_table_name=None, aggregation_keys=None, timestamp_key=None, historical_timestamp_key=None, lookback_window_seconds=None, lookback_window_lag_seconds=0, lookback_count=None, lookback_until_position=0, expression=None):
        return self.client.create_point_in_time_feature(self.feature_group_id, feature_name, history_table_name, aggregation_keys, timestamp_key, historical_timestamp_key, lookback_window_seconds, lookback_window_lag_seconds, lookback_count, lookback_until_position, expression)

    def update_point_in_time_feature(self, feature_name, history_table_name=None, aggregation_keys=None, timestamp_key=None, historical_timestamp_key=None, lookback_window_seconds=None, lookback_window_lag_seconds=0, lookback_count=None, lookback_until_position=0, expression=None, new_feature_name=None):
        return self.client.update_point_in_time_feature(self.feature_group_id, feature_name, history_table_name, aggregation_keys, timestamp_key, historical_timestamp_key, lookback_window_seconds, lookback_window_lag_seconds, lookback_count, lookback_until_position, expression, new_feature_name)

    def set_feature_type(self, feature, feature_type):
        return self.client.set_feature_type(self.feature_group_id, feature, feature_type)

    def invalidate_streaming_data(self, invalid_before_timestamp):
        return self.client.invalidate_streaming_feature_group_data(self.feature_group_id, invalid_before_timestamp)

    def concatenate_data(self, source_feature_group_id, merge_type='UNION', replace_until_timestamp=None):
        return self.client.concatenate_feature_group_data(self.feature_group_id, source_feature_group_id, merge_type, replace_until_timestamp)

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_feature_group(self.feature_group_id)

    def set_indexing_config(self, primary_key=None, update_timestamp_key=None, lookup_keys=None):
        return self.client.set_feature_group_indexing_config(self.feature_group_id, primary_key, update_timestamp_key, lookup_keys)

    def update(self, description=None):
        return self.client.update_feature_group(self.feature_group_id, description)

    def update_sql_definition(self, sql):
        return self.client.update_feature_group_sql_definition(self.feature_group_id, sql)

    def update_function_definition(self, function_source_code=None, function_name=None, input_feature_groups=[]):
        return self.client.update_feature_group_function_definition(self.feature_group_id, function_source_code, function_name, input_feature_groups)

    def update_feature(self, name, select_expression=None, new_name=None):
        return self.client.update_feature(self.feature_group_id, name, select_expression, new_name)

    def list_exports(self):
        return self.client.list_feature_group_exports(self.feature_group_id)

    def set_modifier_lock(self, locked=True):
        return self.client.set_feature_group_modifier_lock(self.feature_group_id, locked)

    def list_modifiers(self):
        return self.client.list_feature_group_modifiers(self.feature_group_id)

    def add_user_to_modifiers(self, email):
        return self.client.add_user_to_feature_group_modifiers(self.feature_group_id, email)

    def add_organization_group_to_modifiers(self, organization_group_id):
        return self.client.add_organization_group_to_feature_group_modifiers(self.feature_group_id, organization_group_id)

    def remove_user_from_modifiers(self, email):
        return self.client.remove_user_from_feature_group_modifiers(self.feature_group_id, email)

    def remove_organization_group_from_modifiers(self, organization_group_id):
        return self.client.remove_organization_group_from_feature_group_modifiers(self.feature_group_id, organization_group_id)

    def delete_feature(self, name):
        return self.client.delete_feature(self.feature_group_id, name)

    def delete(self):
        return self.client.delete_feature_group(self.feature_group_id)

    def create_version(self):
        return self.client.create_feature_group_version(self.feature_group_id)

    def list_versions(self, limit=100, start_after_version=None):
        return self.client.list_feature_group_versions(self.feature_group_id, limit, start_after_version)

    def get_recent_streamed_data(self):
        return self.client.get_recent_feature_group_streamed_data(self.feature_group_id)

    def upsert_data(self, streaming_token, data):
        return self.client.upsert_data(self.feature_group_id, streaming_token, data)

    def append_data(self, streaming_token, data):
        return self.client.append_data(self.feature_group_id, streaming_token, data)

    def wait_for_materialization(self, timeout=7200):
        return self.client._poll(self, {'PENDING', 'GENERATING'}, timeout=timeout)

    def get_status(self):
        return self.describe().latest_feature_group_version.status

    def load_as_pandas(self):
        latest_version = self.describe().latest_feature_group_version
        if not latest_version:
            from client import ApiException
            raise ApiException(409, 'Feature group must first be materialized')
        return latest_version.load_as_pandas()

    def describe_dataset(self):
        if self.dataset_id:
            return self.client.describe_dataset(self.dataset_id)
