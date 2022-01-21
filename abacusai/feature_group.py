from .feature import Feature
from .feature_group_version import FeatureGroupVersion
from .return_class import AbstractApiClass


class FeatureGroup(AbstractApiClass):
    """
        A feature group
    """

    def __init__(self, client, modificationLock=None, featureGroupId=None, name=None, featureGroupSourceType=None, tableName=None, sql=None, datasetId=None, functionSourceCode=None, functionName=None, sourceTables=None, createdAt=None, description=None, featureGroupType=None, useForTraining=None, sqlError=None, latestVersionOutdated=None, tags=None, primaryKey=None, updateTimestampKey=None, lookupKeys=None, featureGroupUse=None, isIncremental=None, mergeConfig=None, features={}, duplicateFeatures={}, latestFeatureGroupVersion={}):
        super().__init__(client, featureGroupId)
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
        self.feature_group_use = featureGroupUse
        self.is_incremental = isIncremental
        self.merge_config = mergeConfig
        self.features = client._build_class(Feature, features)
        self.duplicate_features = client._build_class(
            Feature, duplicateFeatures)
        self.latest_feature_group_version = client._build_class(
            FeatureGroupVersion, latestFeatureGroupVersion)

    def __repr__(self):
        return f"FeatureGroup(modification_lock={repr(self.modification_lock)},\n  feature_group_id={repr(self.feature_group_id)},\n  name={repr(self.name)},\n  feature_group_source_type={repr(self.feature_group_source_type)},\n  table_name={repr(self.table_name)},\n  sql={repr(self.sql)},\n  dataset_id={repr(self.dataset_id)},\n  function_source_code={repr(self.function_source_code)},\n  function_name={repr(self.function_name)},\n  source_tables={repr(self.source_tables)},\n  created_at={repr(self.created_at)},\n  description={repr(self.description)},\n  feature_group_type={repr(self.feature_group_type)},\n  use_for_training={repr(self.use_for_training)},\n  sql_error={repr(self.sql_error)},\n  latest_version_outdated={repr(self.latest_version_outdated)},\n  tags={repr(self.tags)},\n  primary_key={repr(self.primary_key)},\n  update_timestamp_key={repr(self.update_timestamp_key)},\n  lookup_keys={repr(self.lookup_keys)},\n  feature_group_use={repr(self.feature_group_use)},\n  is_incremental={repr(self.is_incremental)},\n  merge_config={repr(self.merge_config)},\n  features={repr(self.features)},\n  duplicate_features={repr(self.duplicate_features)},\n  latest_feature_group_version={repr(self.latest_feature_group_version)})"

    def to_dict(self):
        return {'modification_lock': self.modification_lock, 'feature_group_id': self.feature_group_id, 'name': self.name, 'feature_group_source_type': self.feature_group_source_type, 'table_name': self.table_name, 'sql': self.sql, 'dataset_id': self.dataset_id, 'function_source_code': self.function_source_code, 'function_name': self.function_name, 'source_tables': self.source_tables, 'created_at': self.created_at, 'description': self.description, 'feature_group_type': self.feature_group_type, 'use_for_training': self.use_for_training, 'sql_error': self.sql_error, 'latest_version_outdated': self.latest_version_outdated, 'tags': self.tags, 'primary_key': self.primary_key, 'update_timestamp_key': self.update_timestamp_key, 'lookup_keys': self.lookup_keys, 'feature_group_use': self.feature_group_use, 'is_incremental': self.is_incremental, 'merge_config': self.merge_config, 'features': self._get_attribute_as_dict(self.features), 'duplicate_features': self._get_attribute_as_dict(self.duplicate_features), 'latest_feature_group_version': self._get_attribute_as_dict(self.latest_feature_group_version)}

    def add_to_project(self, project_id, feature_group_type='CUSTOM_TABLE', feature_group_use=None):
        """Adds a feature group to a project,"""
        return self.client.add_feature_group_to_project(self.feature_group_id, project_id, feature_group_type, feature_group_use)

    def remove_from_project(self, project_id):
        """Removes a feature group from a project."""
        return self.client.remove_feature_group_from_project(self.feature_group_id, project_id)

    def set_type(self, project_id, feature_group_type='CUSTOM_TABLE'):
        """Update the feature group type in a project. The feature group must already be added to the project."""
        return self.client.set_feature_group_type(self.feature_group_id, project_id, feature_group_type)

    def use_for_training(self, project_id, use_for_training=True):
        """Use the feature group for model training input"""
        return self.client.use_feature_group_for_training(self.feature_group_id, project_id, use_for_training)

    def create_sampling(self, table_name, sampling_config, description=None):
        """Creates a new feature group defined as a sample of rows from another feature group."""
        return self.client.create_sampling_feature_group(self.feature_group_id, table_name, sampling_config, description)

    def set_sampling_config(self, sampling_config):
        """Set a FeatureGroup’s sampling to the config values provided, so that the rows the FeatureGroup returns will be a sample of those it would otherwise have returned."""
        return self.client.set_feature_group_sampling_config(self.feature_group_id, sampling_config)

    def set_merge_config(self, merge_config):
        """Set a MergeFeatureGroup’s merge config to the values provided, so that the feature group only returns a bounded range of an incremental dataset."""
        return self.client.set_feature_group_merge_config(self.feature_group_id, merge_config)

    def set_schema(self, schema):
        """Creates a new schema and points the feature group to the new feature group schema id."""
        return self.client.set_feature_group_schema(self.feature_group_id, schema)

    def get_schema(self, project_id=None):
        """Returns a schema given a specific FeatureGroup in a project."""
        return self.client.get_feature_group_schema(self.feature_group_id, project_id)

    def create_feature(self, name, select_expression):
        """Creates a new feature in a Feature Group from a SQL select statement"""
        return self.client.create_feature(self.feature_group_id, name, select_expression)

    def add_tag(self, tag):
        """Adds a tag to the feature group"""
        return self.client.add_feature_group_tag(self.feature_group_id, tag)

    def remove_tag(self, tag):
        """Removes a tag from the feature group"""
        return self.client.remove_feature_group_tag(self.feature_group_id, tag)

    def create_nested_feature(self, nested_feature_name, table_name, using_clause, where_clause=None, order_clause=None):
        """Creates a new nested feature in a feature group from a SQL statements to create a new nested feature."""
        return self.client.create_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause)

    def update_nested_feature(self, nested_feature_name, table_name=None, using_clause=None, where_clause=None, order_clause=None, new_nested_feature_name=None):
        """Updates a previously existing nested feature in a feature group."""
        return self.client.update_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause, new_nested_feature_name)

    def delete_nested_feature(self, nested_feature_name):
        """Delete a nested feature."""
        return self.client.delete_nested_feature(self.feature_group_id, nested_feature_name)

    def create_point_in_time_feature(self, feature_name, history_table_name=None, aggregation_keys=None, timestamp_key=None, historical_timestamp_key=None, lookback_window_seconds=None, lookback_window_lag_seconds=0, lookback_count=None, lookback_until_position=0, expression=None):
        """Creates a new point in time feature in a feature group using another historical feature group, window spec and aggregate expression."""
        return self.client.create_point_in_time_feature(self.feature_group_id, feature_name, history_table_name, aggregation_keys, timestamp_key, historical_timestamp_key, lookback_window_seconds, lookback_window_lag_seconds, lookback_count, lookback_until_position, expression)

    def update_point_in_time_feature(self, feature_name, history_table_name=None, aggregation_keys=None, timestamp_key=None, historical_timestamp_key=None, lookback_window_seconds=None, lookback_window_lag_seconds=None, lookback_count=None, lookback_until_position=None, expression=None, new_feature_name=None):
        """Updates an existing point in time feature in a feature group. See createPointInTimeFeature for detailed semantics."""
        return self.client.update_point_in_time_feature(self.feature_group_id, feature_name, history_table_name, aggregation_keys, timestamp_key, historical_timestamp_key, lookback_window_seconds, lookback_window_lag_seconds, lookback_count, lookback_until_position, expression, new_feature_name)

    def set_feature_type(self, feature, feature_type):
        """Set a feature's type in a feature group/. Specify the feature group ID, feature name and feature type, and the method will return the new column with the resulting changes reflected."""
        return self.client.set_feature_type(self.feature_group_id, feature, feature_type)

    def invalidate_streaming_data(self, invalid_before_timestamp):
        """Invalidates all streaming data with timestamp before invalidBeforeTimestamp"""
        return self.client.invalidate_streaming_feature_group_data(self.feature_group_id, invalid_before_timestamp)

    def concatenate_data(self, source_feature_group_id, merge_type='UNION', replace_until_timestamp=None, skip_materialize=False):
        """Concatenates data from one feature group to another. Feature groups can be merged if their schema's are compatible and they have the special updateTimestampKey column and if set, the primaryKey column. The second operand in the concatenate operation will be appended to the first operand (merge target)."""
        return self.client.concatenate_feature_group_data(self.feature_group_id, source_feature_group_id, merge_type, replace_until_timestamp, skip_materialize)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Describe a Feature Group."""
        return self.client.describe_feature_group(self.feature_group_id)

    def set_indexing_config(self, primary_key=None, update_timestamp_key=None, lookup_keys=None):
        """Sets various attributes of the feature group used for deployment lookups and streaming updates."""
        return self.client.set_feature_group_indexing_config(self.feature_group_id, primary_key, update_timestamp_key, lookup_keys)

    def update(self, description=None):
        """Modifies an existing feature group"""
        return self.client.update_feature_group(self.feature_group_id, description)

    def update_sql_definition(self, sql):
        """Updates the SQL statement for a feature group."""
        return self.client.update_feature_group_sql_definition(self.feature_group_id, sql)

    def update_function_definition(self, function_source_code=None, function_name=None, input_feature_groups=None):
        """Updates the function definition for a feature group created using createFeatureGroupFromFunction"""
        return self.client.update_feature_group_function_definition(self.feature_group_id, function_source_code, function_name, input_feature_groups)

    def update_feature(self, name, select_expression=None, new_name=None):
        """Modifies an existing feature in a feature group. A user needs to specify the name and feature group ID and either a SQL statement or new name tp update the feature."""
        return self.client.update_feature(self.feature_group_id, name, select_expression, new_name)

    def list_exports(self):
        """Lists all of the feature group exports for a given feature group"""
        return self.client.list_feature_group_exports(self.feature_group_id)

    def set_modifier_lock(self, locked=True):
        """To lock a feature group to prevent it from being modified."""
        return self.client.set_feature_group_modifier_lock(self.feature_group_id, locked)

    def list_modifiers(self):
        """To list users who can modify a feature group."""
        return self.client.list_feature_group_modifiers(self.feature_group_id)

    def add_user_to_modifiers(self, email):
        """Adds user to a feature group."""
        return self.client.add_user_to_feature_group_modifiers(self.feature_group_id, email)

    def add_organization_group_to_modifiers(self, organization_group_id):
        """Add Organization to a feature group."""
        return self.client.add_organization_group_to_feature_group_modifiers(self.feature_group_id, organization_group_id)

    def remove_user_from_modifiers(self, email):
        """Removes user from a feature group."""
        return self.client.remove_user_from_feature_group_modifiers(self.feature_group_id, email)

    def remove_organization_group_from_modifiers(self, organization_group_id):
        """Removes Organization from a feature group."""
        return self.client.remove_organization_group_from_feature_group_modifiers(self.feature_group_id, organization_group_id)

    def delete_feature(self, name):
        """Removes an existing feature from a feature group. A user needs to specify the name of the feature to be deleted and the feature group ID."""
        return self.client.delete_feature(self.feature_group_id, name)

    def delete(self):
        """Removes an existing feature group."""
        return self.client.delete_feature_group(self.feature_group_id)

    def create_version(self):
        """Creates a snapshot for a specified feature group."""
        return self.client.create_feature_group_version(self.feature_group_id)

    def list_versions(self, limit=100, start_after_version=None):
        """Retrieves a list of all feature group versions for the specified feature group."""
        return self.client.list_feature_group_versions(self.feature_group_id, limit, start_after_version)

    def get_recent_streamed_data(self):
        """Returns recently streamed data to a streaming feature group."""
        return self.client.get_recent_feature_group_streamed_data(self.feature_group_id)

    def upsert_data(self, streaming_token, data):
        """Updates new data into the feature group for a given lookup key recordId."""
        return self.client.upsert_data(self.feature_group_id, streaming_token, data)

    def append_data(self, streaming_token, data):
        """Appends new data into the feature group for a given lookup key recordId."""
        return self.client.append_data(self.feature_group_id, streaming_token, data)

    def wait_for_materialization(self, timeout=7200):
        """
        A waiting call until feature group is materialized.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 7200 milliseconds.

        Returns:
            None
        """
        return self.client._poll(self, {'PENDING', 'GENERATING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the feature group.

        Returns:
            Enum (string): A string describing the status of a feature group (pending, complete, etc.).
        """
        return self.describe().latest_feature_group_version.status

    def load_as_pandas(self):
        """
        Loads the feature groups into a python pandas dataframe.

        Returns:
            DataFrame: A pandas dataframe with annotations and text_snippet columns.
        """
        latest_version = self.describe().latest_feature_group_version
        if not latest_version:
            from .client import ApiException
            raise ApiException(409, 'Feature group must first be materialized')
        return latest_version.load_as_pandas()

    def describe_dataset(self):
        """
        Displays the dataset attached to a feature group.

        Returns:
            Dataset (object): A dataset object with all the relevant information about the dataset.
        """
        if self.dataset_id:
            return self.client.describe_dataset(self.dataset_id)
