from .code_source import CodeSource
from .concatenation_config import ConcatenationConfig
from .feature import Feature
from .feature_group_template import FeatureGroupTemplate
from .feature_group_version import FeatureGroupVersion
from .indexing_config import IndexingConfig
from .point_in_time_group import PointInTimeGroup
from .return_class import AbstractApiClass


class FeatureGroup(AbstractApiClass):
    """


        Args:
            client (ApiClient): An authenticated API Client instance
            modificationLock (bool): 
            featureGroupId (str): 
            name (str): 
            featureGroupSourceType (str): 
            tableName (str): 
            sql (str): 
            datasetId (str): 
            functionSourceCode (str): 
            functionName (str): 
            sourceTables (list): 
            createdAt (str): 
            description (str): 
            featureGroupType (str): 
            sqlError (str): 
            latestVersionOutdated (bool): 
            referencedFeatureGroups (list): 
            tags (list): 
            primaryKey (str): 
            updateTimestampKey (str): 
            lookupKeys (list): 
            streamingEnabled (bool): 
            featureGroupUse (str): 
            incremental (bool): 
            mergeConfig (dict): 
            transformConfig (dict): 
            samplingConfig (dict): 
            cpuSize (str): 
            memory (number(integer)): 
            streamingReady (bool): 
            featureTags (dict): 
            moduleName (str): 
            templateBindings (list): 
            featureExpression (str): 
            useOriginalCsvNames (bool): 
            pythonFunctionBindings (list): 
            pythonFunctionName (str): 
            annotationConfig (dict): 
            features (Feature): 
            duplicateFeatures (Feature): 
            pointInTimeGroups (PointInTimeGroup): 
            latestFeatureGroupVersion (FeatureGroupVersion): 
            concatenationConfig (ConcatenationConfig): 
            indexingConfig (IndexingConfig): 
            codeSource (CodeSource): 
            featureGroupTemplate (FeatureGroupTemplate): 
    """

    def __init__(self, client, modificationLock=None, featureGroupId=None, name=None, featureGroupSourceType=None, tableName=None, sql=None, datasetId=None, functionSourceCode=None, functionName=None, sourceTables=None, createdAt=None, description=None, featureGroupType=None, sqlError=None, latestVersionOutdated=None, referencedFeatureGroups=None, tags=None, primaryKey=None, updateTimestampKey=None, lookupKeys=None, streamingEnabled=None, featureGroupUse=None, incremental=None, mergeConfig=None, transformConfig=None, samplingConfig=None, cpuSize=None, memory=None, streamingReady=None, featureTags=None, moduleName=None, templateBindings=None, featureExpression=None, useOriginalCsvNames=None, pythonFunctionBindings=None, pythonFunctionName=None, annotationConfig=None, features={}, duplicateFeatures={}, pointInTimeGroups={}, concatenationConfig={}, indexingConfig={}, codeSource={}, featureGroupTemplate={}, latestFeatureGroupVersion={}):
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
        self.sql_error = sqlError
        self.latest_version_outdated = latestVersionOutdated
        self.referenced_feature_groups = referencedFeatureGroups
        self.tags = tags
        self.primary_key = primaryKey
        self.update_timestamp_key = updateTimestampKey
        self.lookup_keys = lookupKeys
        self.streaming_enabled = streamingEnabled
        self.feature_group_use = featureGroupUse
        self.incremental = incremental
        self.merge_config = mergeConfig
        self.transform_config = transformConfig
        self.sampling_config = samplingConfig
        self.cpu_size = cpuSize
        self.memory = memory
        self.streaming_ready = streamingReady
        self.feature_tags = featureTags
        self.module_name = moduleName
        self.template_bindings = templateBindings
        self.feature_expression = featureExpression
        self.use_original_csv_names = useOriginalCsvNames
        self.python_function_bindings = pythonFunctionBindings
        self.python_function_name = pythonFunctionName
        self.annotation_config = annotationConfig
        self.features = client._build_class(Feature, features)
        self.duplicate_features = client._build_class(
            Feature, duplicateFeatures)
        self.point_in_time_groups = client._build_class(
            PointInTimeGroup, pointInTimeGroups)
        self.concatenation_config = client._build_class(
            ConcatenationConfig, concatenationConfig)
        self.indexing_config = client._build_class(
            IndexingConfig, indexingConfig)
        self.code_source = client._build_class(CodeSource, codeSource)
        self.feature_group_template = client._build_class(
            FeatureGroupTemplate, featureGroupTemplate)
        self.latest_feature_group_version = client._build_class(
            FeatureGroupVersion, latestFeatureGroupVersion)

    def __repr__(self):
        return f"FeatureGroup(modification_lock={repr(self.modification_lock)},\n  feature_group_id={repr(self.feature_group_id)},\n  name={repr(self.name)},\n  feature_group_source_type={repr(self.feature_group_source_type)},\n  table_name={repr(self.table_name)},\n  sql={repr(self.sql)},\n  dataset_id={repr(self.dataset_id)},\n  function_source_code={repr(self.function_source_code)},\n  function_name={repr(self.function_name)},\n  source_tables={repr(self.source_tables)},\n  created_at={repr(self.created_at)},\n  description={repr(self.description)},\n  feature_group_type={repr(self.feature_group_type)},\n  sql_error={repr(self.sql_error)},\n  latest_version_outdated={repr(self.latest_version_outdated)},\n  referenced_feature_groups={repr(self.referenced_feature_groups)},\n  tags={repr(self.tags)},\n  primary_key={repr(self.primary_key)},\n  update_timestamp_key={repr(self.update_timestamp_key)},\n  lookup_keys={repr(self.lookup_keys)},\n  streaming_enabled={repr(self.streaming_enabled)},\n  feature_group_use={repr(self.feature_group_use)},\n  incremental={repr(self.incremental)},\n  merge_config={repr(self.merge_config)},\n  transform_config={repr(self.transform_config)},\n  sampling_config={repr(self.sampling_config)},\n  cpu_size={repr(self.cpu_size)},\n  memory={repr(self.memory)},\n  streaming_ready={repr(self.streaming_ready)},\n  feature_tags={repr(self.feature_tags)},\n  module_name={repr(self.module_name)},\n  template_bindings={repr(self.template_bindings)},\n  feature_expression={repr(self.feature_expression)},\n  use_original_csv_names={repr(self.use_original_csv_names)},\n  python_function_bindings={repr(self.python_function_bindings)},\n  python_function_name={repr(self.python_function_name)},\n  annotation_config={repr(self.annotation_config)},\n  features={repr(self.features)},\n  duplicate_features={repr(self.duplicate_features)},\n  point_in_time_groups={repr(self.point_in_time_groups)},\n  concatenation_config={repr(self.concatenation_config)},\n  indexing_config={repr(self.indexing_config)},\n  code_source={repr(self.code_source)},\n  feature_group_template={repr(self.feature_group_template)},\n  latest_feature_group_version={repr(self.latest_feature_group_version)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'modification_lock': self.modification_lock, 'feature_group_id': self.feature_group_id, 'name': self.name, 'feature_group_source_type': self.feature_group_source_type, 'table_name': self.table_name, 'sql': self.sql, 'dataset_id': self.dataset_id, 'function_source_code': self.function_source_code, 'function_name': self.function_name, 'source_tables': self.source_tables, 'created_at': self.created_at, 'description': self.description, 'feature_group_type': self.feature_group_type, 'sql_error': self.sql_error, 'latest_version_outdated': self.latest_version_outdated, 'referenced_feature_groups': self.referenced_feature_groups, 'tags': self.tags, 'primary_key': self.primary_key, 'update_timestamp_key': self.update_timestamp_key, 'lookup_keys': self.lookup_keys, 'streaming_enabled': self.streaming_enabled, 'feature_group_use': self.feature_group_use, 'incremental': self.incremental, 'merge_config': self.merge_config, 'transform_config': self.transform_config, 'sampling_config': self.sampling_config, 'cpu_size': self.cpu_size, 'memory': self.memory, 'streaming_ready': self.streaming_ready, 'feature_tags': self.feature_tags, 'module_name': self.module_name, 'template_bindings': self.template_bindings, 'feature_expression': self.feature_expression, 'use_original_csv_names': self.use_original_csv_names, 'python_function_bindings': self.python_function_bindings, 'python_function_name': self.python_function_name, 'annotation_config': self.annotation_config, 'features': self._get_attribute_as_dict(self.features), 'duplicate_features': self._get_attribute_as_dict(self.duplicate_features), 'point_in_time_groups': self._get_attribute_as_dict(self.point_in_time_groups), 'concatenation_config': self._get_attribute_as_dict(self.concatenation_config), 'indexing_config': self._get_attribute_as_dict(self.indexing_config), 'code_source': self._get_attribute_as_dict(self.code_source), 'feature_group_template': self._get_attribute_as_dict(self.feature_group_template), 'latest_feature_group_version': self._get_attribute_as_dict(self.latest_feature_group_version)}

    def add_to_project(self, project_id: str, feature_group_type: str = 'CUSTOM_TABLE', feature_group_use: str = None):
        """
        Adds a feature group to a project,

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_type (str):  The feature group type of the feature group. The type is based on the use case under which the feature group is being created. For example, Catalog Attributes can be a feature group type under personalized recommendation use case.
            feature_group_use (str): The user assigned feature group use which allows for organizing project feature groups  DATA_WRANGLING,  TRAINING_INPUT,  BATCH_PREDICTION_INPUT
        """
        return self.client.add_feature_group_to_project(self.feature_group_id, project_id, feature_group_type, feature_group_use)

    def remove_from_project(self, project_id: str):
        """
        Removes a feature group from a project.

        Args:
            project_id (str): The unique ID associated with the project.
        """
        return self.client.remove_feature_group_from_project(self.feature_group_id, project_id)

    def set_type(self, project_id: str, feature_group_type: str = 'CUSTOM_TABLE'):
        """
        Update the feature group type in a project. The feature group must already be added to the project.

        Args:
            project_id (str): The unique ID associated with the project.
            feature_group_type (str): The feature group type to set the feature group as. The type is based on the use case under which the feature group is being created. For example, Catalog Attributes can be a feature group type under personalized recommendation use case.
        """
        return self.client.set_feature_group_type(self.feature_group_id, project_id, feature_group_type)

    def use_for_training(self, project_id: str, use_for_training: bool = True):
        """
        Use the feature group for model training input

        Args:
            project_id (str): The unique ID associated with the project.
            use_for_training (bool): Boolean variable to include or exclude a feature group from a model's training. Only one feature group per type can be used for training
        """
        return self.client.use_feature_group_for_training(self.feature_group_id, project_id, use_for_training)

    def describe_annotation(self, feature_name: str = None, doc_id: str = None, feature_group_row_identifier: str = None):
        """
        Get the latest annotation entry for a given feature group, feature, and document.

        Args:
            feature_name (str): The name of the feature the annotation is on.
            doc_id (str): The ID of the primary document the annotation is on.
            feature_group_row_identifier (str): The key value of the feature group row the annotation is on (cast to string). Usually the primary key value. At least one of the doc_id or key value must be provided so that the correct annotation can be identified.

        Returns:
            AnnotationEntry: The latest annotation entry for the given feature group, feature, and document and/or annotation key value
        """
        return self.client.describe_annotation(self.feature_group_id, feature_name, doc_id, feature_group_row_identifier)

    def create_sampling(self, table_name: str, sampling_config: dict, description: str = None):
        """
        Creates a new feature group defined as a sample of rows from another feature group.

        For efficiency, sampling is approximate unless otherwise specified. (E.g. the number of rows may vary slightly from what was requested).


        Args:
            table_name (str): The unique name to be given to this sampling feature group.
            sampling_config (dict): JSON object (aka map) defining the sampling method and its parameters.
            description (str): A human-readable description of this feature group.

        Returns:
            FeatureGroup: The created feature group.
        """
        return self.client.create_sampling_feature_group(self.feature_group_id, table_name, sampling_config, description)

    def set_sampling_config(self, sampling_config: dict):
        """
        Set a FeatureGroup’s sampling to the config values provided, so that the rows the FeatureGroup returns will be a sample of those it would otherwise have returned.

        Currently, sampling is only for Sampling FeatureGroups, so this API only allows calling on that kind of FeatureGroup.


        Args:
            sampling_config (dict): A json object string specifying the sampling method and parameters specific to that sampling method. Empty sampling_config means no sampling.

        Returns:
            FeatureGroup: The updated feature group.
        """
        return self.client.set_feature_group_sampling_config(self.feature_group_id, sampling_config)

    def set_merge_config(self, merge_config: dict):
        """
        Set a MergeFeatureGroup’s merge config to the values provided, so that the feature group only returns a bounded range of an incremental dataset.

        Args:
            merge_config (dict): A json object string specifying the merge rule. An empty mergeConfig will default to only including the latest Dataset Version.
        """
        return self.client.set_feature_group_merge_config(self.feature_group_id, merge_config)

    def set_transform_config(self, transform_config: dict):
        """
        Set a TransformFeatureGroup’s transform config to the values provided.

        Args:
            transform_config (dict): A json object string specifying the pre-defined transformation.
        """
        return self.client.set_feature_group_transform_config(self.feature_group_id, transform_config)

    def set_schema(self, schema: list):
        """
        Creates a new schema and points the feature group to the new feature group schema id.

        Args:
            schema (list): An array of json objects with 'name' and 'dataType' properties.
        """
        return self.client.set_feature_group_schema(self.feature_group_id, schema)

    def get_schema(self, project_id: str = None):
        """
        Returns a schema given a specific FeatureGroup in a project.

        Args:
            project_id (str): The unique ID associated with the project.

        Returns:
            Feature: An array of objects for each column in the specified feature group.
        """
        return self.client.get_feature_group_schema(self.feature_group_id, project_id)

    def create_feature(self, name: str, select_expression: str):
        """
        Creates a new feature in a Feature Group from a SQL select statement

        Args:
            name (str): The name of the feature to add
            select_expression (str): SQL select expression to create the feature

        Returns:
            FeatureGroup: A feature group object with the newly added feature.
        """
        return self.client.create_feature(self.feature_group_id, name, select_expression)

    def add_tag(self, tag: str):
        """
        Adds a tag to the feature group

        Args:
            tag (str): The tag to add to the feature group
        """
        return self.client.add_feature_group_tag(self.feature_group_id, tag)

    def remove_tag(self, tag: str):
        """
        Removes a tag from the feature group

        Args:
            tag (str): The tag to add to the feature group
        """
        return self.client.remove_feature_group_tag(self.feature_group_id, tag)

    def add_annotatable_feature(self, name: str, annotation_type: str):
        """


        Args:
            name (str): 
            annotation_type (str): 

        Returns:
            FeatureGroup: None
        """
        return self.client.add_annotatable_feature(self.feature_group_id, name, annotation_type)

    def set_feature_as_annotatable_feature(self, feature_name: str, annotation_type: str, feature_group_row_identifier_feature: str = None, doc_id_feature: str = None):
        """


        Args:
            feature_name (str): 
            annotation_type (str): 
            feature_group_row_identifier_feature (str): 
            doc_id_feature (str): 

        Returns:
            FeatureGroup: None
        """
        return self.client.set_feature_as_annotatable_feature(self.feature_group_id, feature_name, annotation_type, feature_group_row_identifier_feature, doc_id_feature)

    def unset_feature_as_annotatable_feature(self, feature_name: str):
        """


        Args:
            feature_name (str): 

        Returns:
            FeatureGroup: None
        """
        return self.client.unset_feature_as_annotatable_feature(self.feature_group_id, feature_name)

    def add_annotation_label(self, label_name: str, annotation_type: str, label_definition: str = None):
        """


        Args:
            label_name (str): 
            annotation_type (str): 
            label_definition (str): 

        Returns:
            FeatureGroup: None
        """
        return self.client.add_feature_group_annotation_label(self.feature_group_id, label_name, annotation_type, label_definition)

    def remove_annotation_label(self, label_name: str):
        """


        Args:
            label_name (str): 

        Returns:
            FeatureGroup: None
        """
        return self.client.remove_feature_group_annotation_label(self.feature_group_id, label_name)

    def add_feature_tag(self, feature: str, tag: str):
        """


        Args:
            feature (str): 
            tag (str): 
        """
        return self.client.add_feature_tag(self.feature_group_id, feature, tag)

    def remove_feature_tag(self, feature: str, tag: str):
        """


        Args:
            feature (str): 
            tag (str): 
        """
        return self.client.remove_feature_tag(self.feature_group_id, feature, tag)

    def create_nested_feature(self, nested_feature_name: str, table_name: str, using_clause: str, where_clause: str = None, order_clause: str = None):
        """
        Creates a new nested feature in a feature group from a SQL statements to create a new nested feature.

        Args:
            nested_feature_name (str): The name of the feature.
            table_name (str): The table name of the feature group to nest
            using_clause (str): The SQL join column or logic to join the nested table with the parent
            where_clause (str): A SQL where statement to filter the nested rows
            order_clause (str): A SQL clause to order the nested rows

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature.
        """
        return self.client.create_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause)

    def update_nested_feature(self, nested_feature_name: str, table_name: str = None, using_clause: str = None, where_clause: str = None, order_clause: str = None, new_nested_feature_name: str = None):
        """
        Updates a previously existing nested feature in a feature group.

        Args:
            nested_feature_name (str): The name of the feature to be updated.
            table_name (str): The name of the table.
            using_clause (str): The SQL join column or logic to join the nested table with the parent
            where_clause (str): A SQL where statement to filter the nested rows
            order_clause (str): A SQL clause to order the nested rows
            new_nested_feature_name (str): New name for the nested feature.

        Returns:
            FeatureGroup: A feature group object with the updated nested feature.
        """
        return self.client.update_nested_feature(self.feature_group_id, nested_feature_name, table_name, using_clause, where_clause, order_clause, new_nested_feature_name)

    def delete_nested_feature(self, nested_feature_name: str):
        """
        Delete a nested feature.

        Args:
            nested_feature_name (str): The name of the feature to be updated.

        Returns:
            FeatureGroup: A feature group object without the deleted nested feature.
        """
        return self.client.delete_nested_feature(self.feature_group_id, nested_feature_name)

    def create_point_in_time_feature(self, feature_name: str, history_table_name: str, aggregation_keys: list, timestamp_key: str, historical_timestamp_key: str, expression: str, lookback_window_seconds: float = None, lookback_window_lag_seconds: float = 0, lookback_count: int = None, lookback_until_position: int = 0):
        """
        Creates a new point in time feature in a feature group using another historical feature group, window spec and aggregate expression.

        We use the aggregation keys, and either the lookbackWindowSeconds or the lookbackCount values to perform the window aggregation for every row in the current feature group.
        If the window is specified in seconds, then all rows in the history table which match the aggregation keys and with historicalTimeFeature >= lookbackStartCount and < the value
        of the current rows timeFeature are considered. An option lookbackWindowLagSeconds (+ve or -ve) can be used to offset the current value of the timeFeature. If this value
        is negative, we will look at the future rows in the history table, so care must be taken to make sure that these rows are available in the online context when we are performing
        a lookup on this feature group. If window is specified in counts, then we order the historical table rows aligning by time and consider rows from the window where
        the rank order is >= lookbackCount and includes the row just prior to the current one. The lag is specified in term of positions using lookbackUntilPosition.


        Args:
            feature_name (str): The name of the feature to create
            history_table_name (str): The table name of the history table.
            aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation.
            timestamp_key (str): Name of feature which contains the timestamp value for the point in time feature
            historical_timestamp_key (str): Name of feature which contains the historical timestamp.
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
            lookback_window_seconds (float): If window is specified in terms of time, number of seconds in the past from the current time for start of the window.
            lookback_window_lag_seconds (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature.
        """
        return self.client.create_point_in_time_feature(self.feature_group_id, feature_name, history_table_name, aggregation_keys, timestamp_key, historical_timestamp_key, expression, lookback_window_seconds, lookback_window_lag_seconds, lookback_count, lookback_until_position)

    def update_point_in_time_feature(self, feature_name: str, history_table_name: str = None, aggregation_keys: list = None, timestamp_key: str = None, historical_timestamp_key: str = None, expression: str = None, lookback_window_seconds: float = None, lookback_window_lag_seconds: float = None, lookback_count: int = None, lookback_until_position: int = None, new_feature_name: str = None):
        """
        Updates an existing point in time feature in a feature group. See createPointInTimeFeature for detailed semantics.

        Args:
            feature_name (str): The name of the feature.
            history_table_name (str): The table name of the history table. If not specified, we use the current table to do a self join.
            aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation.
            timestamp_key (str): Name of feature which contains the timestamp value for the point in time feature
            historical_timestamp_key (str): Name of feature which contains the historical timestamp.
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.
            lookback_window_seconds (float): If window is specified in terms of time, number of seconds in the past from the current time for start of the window.
            lookback_window_lag_seconds (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.
            new_feature_name (str): New name for the point in time feature.

        Returns:
            FeatureGroup: A feature group object with the newly added nested feature.
        """
        return self.client.update_point_in_time_feature(self.feature_group_id, feature_name, history_table_name, aggregation_keys, timestamp_key, historical_timestamp_key, expression, lookback_window_seconds, lookback_window_lag_seconds, lookback_count, lookback_until_position, new_feature_name)

    def create_point_in_time_group(self, group_name: str, window_key: str, aggregation_keys: list, history_table_name: str = None, history_window_key: str = None, history_aggregation_keys: list = None, lookback_window: float = None, lookback_window_lag: float = 0, lookback_count: int = None, lookback_until_position: int = 0):
        """
        Create point in time group

        Args:
            group_name (str): The name of the point in time group
            window_key (str): Name of feature to use for ordering the rows on the source table
            aggregation_keys (list): List of keys to perform on the source table for the window aggregation.
            history_table_name (str): The table to use for aggregating, if not provided, the source table will be used
            history_window_key (str): Name of feature to use for ordering the rows on the history table. If not provided, the windowKey from the source table will be used
            history_aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation. If not provided, the aggregationKeys from the source table will be used. Must be the same length and order as the source table's aggregationKeys
            lookback_window (float): Number of seconds in the past from the current time for start of the window. If 0, the lookback will include all rows.
            lookback_window_lag (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.

        Returns:
            FeatureGroup: The feature group after the point in time group has been created
        """
        return self.client.create_point_in_time_group(self.feature_group_id, group_name, window_key, aggregation_keys, history_table_name, history_window_key, history_aggregation_keys, lookback_window, lookback_window_lag, lookback_count, lookback_until_position)

    def update_point_in_time_group(self, group_name: str, window_key: str = None, aggregation_keys: list = None, history_table_name: str = None, history_window_key: str = None, history_aggregation_keys: list = None, lookback_window: float = None, lookback_window_lag: float = None, lookback_count: int = None, lookback_until_position: int = None):
        """
        Update point in time group

        Args:
            group_name (str): The name of the point in time group
            window_key (str): Name of feature which contains the timestamp value for the point in time feature
            aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation.
            history_table_name (str): The table to use for aggregating, if not provided, the source table will be used
            history_window_key (str): Name of feature to use for ordering the rows on the history table. If not provided, the windowKey from the source table will be used
            history_aggregation_keys (list): List of keys to use for join the historical table and performing the window aggregation. If not provided, the aggregationKeys from the source table will be used. Must be the same length and order as the source table's aggregationKeys
            lookback_window (float): Number of seconds in the past from the current time for start of the window.
            lookback_window_lag (float): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window. If it is negative, we are looking at the "future" rows in the history table.
            lookback_count (int): If window is specified in terms of count, the start position of the window (0 is the current row)
            lookback_until_position (int): Optional lag to offset the closest point for the window. If it is positive, we delay the start of window by that many rows. If it is negative, we are looking at those many "future" rows in the history table.

        Returns:
            FeatureGroup: The feature group after the update has been applied
        """
        return self.client.update_point_in_time_group(self.feature_group_id, group_name, window_key, aggregation_keys, history_table_name, history_window_key, history_aggregation_keys, lookback_window, lookback_window_lag, lookback_count, lookback_until_position)

    def delete_point_in_time_group(self, group_name: str):
        """
        Delete point in time group

        Args:
            group_name (str): The name of the point in time group

        Returns:
            FeatureGroup: The feature group after the point in time group has been deleted
        """
        return self.client.delete_point_in_time_group(self.feature_group_id, group_name)

    def create_point_in_time_group_feature(self, group_name: str, name: str, expression: str):
        """
        Create point in time group feature

        Args:
            group_name (str): The name of the point in time group
            name (str): The name of the feature to add to the point in time group
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.

        Returns:
            FeatureGroup: The feature group after the update has been applied
        """
        return self.client.create_point_in_time_group_feature(self.feature_group_id, group_name, name, expression)

    def update_point_in_time_group_feature(self, group_name: str, name: str, expression: str):
        """
        Update a feature's SQL expression in a point in time group

        Args:
            group_name (str): The name of the point in time group
            name (str): The name of the feature to add to the point in time group
            expression (str): SQL Aggregate expression which can convert a sequence of rows into a scalar value.

        Returns:
            FeatureGroup: The feature group after the update has been applied
        """
        return self.client.update_point_in_time_group_feature(self.feature_group_id, group_name, name, expression)

    def set_feature_type(self, feature: str, feature_type: str):
        """
        Set a feature's type in a feature group/. Specify the feature group ID, feature name and feature type, and the method will return the new column with the resulting changes reflected.

        Args:
            feature (str): The name of the feature.
            feature_type (str): The machine learning type of the data in the feature.  CATEGORICAL,  CATEGORICAL_LIST,  NUMERICAL,  TIMESTAMP,  TEXT,  EMAIL,  LABEL_LIST,  JSON,  OBJECT_REFERENCE,  MULTICATEGORICAL_LIST,  COORDINATE_LIST,  NUMERICAL_LIST,  TIMESTAMP_LIST Refer to the (guide on feature types)[https://api.abacus.ai/app/help/class/FeatureType] for more information. Note: Some FeatureMappings will restrict the options or explicitly set the FeatureType.

        Returns:
            Schema: The feature group after the data_type is applied
        """
        return self.client.set_feature_type(self.feature_group_id, feature, feature_type)

    def invalidate_streaming_data(self, invalid_before_timestamp: int):
        """
        Invalidates all streaming data with timestamp before invalidBeforeTimestamp

        Args:
            invalid_before_timestamp (int): The unix timestamp, any data which has a timestamp before this time will be deleted
        """
        return self.client.invalidate_streaming_feature_group_data(self.feature_group_id, invalid_before_timestamp)

    def concatenate_data(self, source_feature_group_id: str, merge_type: str = 'UNION', replace_until_timestamp: int = None, skip_materialize: bool = False):
        """
        Concatenates data from one feature group to another. Feature groups can be merged if their schema's are compatible and they have the special updateTimestampKey column and if set, the primaryKey column. The second operand in the concatenate operation will be appended to the first operand (merge target).

        Args:
            source_feature_group_id (str): The feature group to concatenate with the destination feature group.
            merge_type (str): UNION or INTERSECTION
            replace_until_timestamp (int): The unix timestamp to specify the point till which we will replace data from the source feature group.
            skip_materialize (bool): If true, will not materialize the concatenated feature group
        """
        return self.client.concatenate_feature_group_data(self.feature_group_id, source_feature_group_id, merge_type, replace_until_timestamp, skip_materialize)

    def remove_concatenation_config(self):
        """
        Removes the concatenation config on a destination feature group.

        Args:
            feature_group_id (str): Removes the concatenation configuration on a destination feature group
        """
        return self.client.remove_concatenation_config(self.feature_group_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            FeatureGroup: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describe a Feature Group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            FeatureGroup: The feature group object.
        """
        return self.client.describe_feature_group(self.feature_group_id)

    def set_indexing_config(self, primary_key: str = None, update_timestamp_key: str = None, lookup_keys: list = None):
        """
        Sets various attributes of the feature group used for deployment lookups and streaming updates.

        Args:
            primary_key (str): Name of feature which defines the primary key of the feature group.
            update_timestamp_key (str): Name of feature which defines the update timestamp of the feature group - used in concatenation and primary key deduplication.
            lookup_keys (list): List of feature names which can be used in the lookup api to restrict the computation to a set of dataset rows. These feature names have to correspond to underlying dataset columns.
        """
        return self.client.set_feature_group_indexing_config(self.feature_group_id, primary_key, update_timestamp_key, lookup_keys)

    def update(self, description: str = None):
        """
        Modifies an existing feature group

        Args:
            description (str): The description about the feature group.

        Returns:
            FeatureGroup: The updated feature group object.
        """
        return self.client.update_feature_group(self.feature_group_id, description)

    def detach_from_template(self):
        """
        Update a feature group to detach it from a template.

        Currently, this converts the feature group into a SQL feature group rather than a template feature group.


        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            FeatureGroup: The updated feature group
        """
        return self.client.detach_feature_group_from_template(self.feature_group_id)

    def update_template_bindings(self, template_bindings: list = None):
        """
        Update the feature group template bindings for a template feature group.

        Args:
            template_bindings (list): Values in these bindings override values set in the template.

        Returns:
            FeatureGroup: The updated feature group
        """
        return self.client.update_feature_group_template_bindings(self.feature_group_id, template_bindings)

    def update_python_function_bindings(self, python_function_bindings: list):
        """
        Updates an existing Feature Group's python function bindings from a user provided Python Function. If a list of feature groups are supplied within the python function

        bindings, we will provide as arguments to the function DataFrame's (pandas in the case of Python) with the materialized
        feature groups for those input feature groups.


        Args:
            python_function_bindings (list): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].
        """
        return self.client.update_feature_group_python_function_bindings(self.feature_group_id, python_function_bindings)

    def update_sql_definition(self, sql: str):
        """
        Updates the SQL statement for a feature group.

        Args:
            sql (str): Input SQL statement for the feature group.

        Returns:
            FeatureGroup: The updated feature group
        """
        return self.client.update_feature_group_sql_definition(self.feature_group_id, sql)

    def update_dataset_feature_expression(self, feature_expression: str):
        """
        Updates the SQL feature expression for a dataset feature group's custom features

        Args:
            feature_expression (str): Input SQL statement for the feature group.

        Returns:
            FeatureGroup: The updated feature group
        """
        return self.client.update_dataset_feature_group_feature_expression(self.feature_group_id, feature_expression)

    def update_function_definition(self, function_source_code: str = None, function_name: str = None, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None, use_original_csv_names: bool = False, python_function_bindings: list = None):
        """
        Updates the function definition for a feature group created using createFeatureGroupFromFunction

        Args:
            function_source_code (str): Contents of a valid source code file in a supported Feature Group specification language (currently only Python). The source code should contain a function called function_name. A list of allowed import and system libraries for each language is specified in the user functions documentation section.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency
            use_original_csv_names (bool): If set to true, feature group uses the original column names for input feature groups from csv datasets.
            python_function_bindings (list): python_function_bindings (List[Python Function Arguments]): List of arguments to be supplied to the function as parameters in the format [{'name': 'function_argument', 'variable_type': 'FEATURE_GROUP', 'value': 'name_of_feature_group'}].

        Returns:
            FeatureGroup: The updated feature group
        """
        return self.client.update_feature_group_function_definition(self.feature_group_id, function_source_code, function_name, input_feature_groups, cpu_size, memory, package_requirements, use_original_csv_names, python_function_bindings)

    def update_zip(self, function_name: str, module_name: str, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None):
        """
        Updates the zip for a feature group created using createFeatureGroupFromZip

        Args:
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            Upload: The Upload to upload the zip file to
        """
        return self.client.update_feature_group_zip(self.feature_group_id, function_name, module_name, input_feature_groups, cpu_size, memory, package_requirements)

    def update_git(self, application_connector_id: str = None, branch_name: str = None, python_root: str = None, function_name: str = None, module_name: str = None, input_feature_groups: list = None, cpu_size: str = None, memory: int = None, package_requirements: dict = None):
        """
        Updates a feature group created using createFeatureGroupFromGit

        Args:
            application_connector_id (str): The unique ID associated with the git application connector.
            branch_name (str): Name of the branch in the git repository to be used for training.
            python_root (str): Path from the top level of the git repository to the directory containing the Python source code. If not provided, the default is the root of the git repository.
            function_name (str): Name of the function found in the source code that will be executed (on the optional inputs) to materialize this feature group.
            module_name (str): Path to the file with the feature group function.
            input_feature_groups (list): List of feature groups that are supplied to the function as parameters. Each of the parameters are materialized Dataframes (same type as the functions return value).
            cpu_size (str): Size of the cpu for the feature group function
            memory (int): Memory (in GB) for the feature group function
            package_requirements (dict): Json with key value pairs corresponding to package: version for each dependency

        Returns:
            FeatureGroup: The updated FeatureGroup
        """
        return self.client.update_feature_group_git(self.feature_group_id, application_connector_id, branch_name, python_root, function_name, module_name, input_feature_groups, cpu_size, memory, package_requirements)

    def update_feature(self, name: str, select_expression: str = None, new_name: str = None):
        """
        Modifies an existing feature in a feature group. A user needs to specify the name and feature group ID and either a SQL statement or new name to update the feature.

        Args:
            name (str): The name of the feature to be updated.
            select_expression (str): Input SQL statement for modifying the feature.
            new_name (str):  The new name of the feature.

        Returns:
            FeatureGroup: The updated feature group object.
        """
        return self.client.update_feature(self.feature_group_id, name, select_expression, new_name)

    def list_exports(self):
        """
        Lists all of the feature group exports for a given feature group

        Args:
            feature_group_id (str): The ID of the feature group

        Returns:
            FeatureGroupExport: The feature group exports
        """
        return self.client.list_feature_group_exports(self.feature_group_id)

    def set_modifier_lock(self, locked: bool = True):
        """
        To lock a feature group to prevent it from being modified.

        Args:
            locked (bool): True or False to disable or enable feature group modification.
        """
        return self.client.set_feature_group_modifier_lock(self.feature_group_id, locked)

    def list_modifiers(self):
        """
        To list users who can modify a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.

        Returns:
            ModificationLockInfo: Modification lock status and groups and organizations added to the feature group.
        """
        return self.client.list_feature_group_modifiers(self.feature_group_id)

    def add_user_to_modifiers(self, email: str):
        """
        Adds user to a feature group.

        Args:
            email (str): The email address of the user to be removed.
        """
        return self.client.add_user_to_feature_group_modifiers(self.feature_group_id, email)

    def add_organization_group_to_modifiers(self, organization_group_id: str):
        """
        Add Organization to a feature group.

        Args:
            organization_group_id (str): The unique ID associated with the organization group.
        """
        return self.client.add_organization_group_to_feature_group_modifiers(self.feature_group_id, organization_group_id)

    def remove_user_from_modifiers(self, email: str):
        """
        Removes user from a feature group.

        Args:
            email (str): The email address of the user to be removed.
        """
        return self.client.remove_user_from_feature_group_modifiers(self.feature_group_id, email)

    def remove_organization_group_from_modifiers(self, organization_group_id: str):
        """
        Removes Organization from a feature group.

        Args:
            organization_group_id (str): The unique ID associated with the organization group.
        """
        return self.client.remove_organization_group_from_feature_group_modifiers(self.feature_group_id, organization_group_id)

    def delete_feature(self, name: str):
        """
        Removes an existing feature from a feature group. A user needs to specify the name of the feature to be deleted and the feature group ID.

        Args:
            name (str): The name of the feature to be deleted.

        Returns:
            FeatureGroup: The updated feature group object.
        """
        return self.client.delete_feature(self.feature_group_id, name)

    def delete(self):
        """
        Removes an existing feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
        """
        return self.client.delete_feature_group(self.feature_group_id)

    def create_version(self, variable_bindings: dict = None):
        """
        Creates a snapshot for a specified feature group.

        Args:
            variable_bindings (dict): (JSON Object): JSON object (aka map) defining variable bindings that override parent feature group values.

        Returns:
            FeatureGroupVersion: A feature group version.
        """
        return self.client.create_feature_group_version(self.feature_group_id, variable_bindings)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of all feature group versions for the specified feature group.

        Args:
            limit (int): The max length of the returned versions
            start_after_version (str): Results will start after this version

        Returns:
            FeatureGroupVersion: An array of feature group version.
        """
        return self.client.list_feature_group_versions(self.feature_group_id, limit, start_after_version)

    def create_template(self, name: str, template_sql: str, template_variables: list, description: str = None, template_bindings: list = None, should_attach_feature_group_to_template: bool = False):
        """
        Create a feature group template.

        Args:
            name (str): The user-friendly of for this feature group template.
            template_sql (str): The template sql that will be resolved by applying values from the template variables to generate sql for a feature group.
            template_variables (list): The template variables for resolving the template.
            description (str): A description of this feature group template
            template_bindings (list): If the feature group will be attached to the newly created template, set these variable bindings on that feature group.
            should_attach_feature_group_to_template (bool): Set to True to convert the feature group to a template feature group and attach it to the newly created template.

        Returns:
            FeatureGroupTemplate: The created feature group template
        """
        return self.client.create_feature_group_template(self.feature_group_id, name, template_sql, template_variables, description, template_bindings, should_attach_feature_group_to_template)

    def suggest_template_for(self):
        """
        Suggest values for a feature gruop template, based on a feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group to use for suggesting values to use for the template.

        Returns:
            FeatureGroupTemplate: None
        """
        return self.client.suggest_feature_group_template_for_feature_group(self.feature_group_id)

    def get_recent_streamed_data(self):
        """
        Returns recently streamed data to a streaming feature group.

        Args:
            feature_group_id (str): The unique ID associated with the feature group.
        """
        return self.client.get_recent_feature_group_streamed_data(self.feature_group_id)

    def create_prediction_metric(self, prediction_metric_config: dict, project_id: str = None):
        """
        Create a prediction metric job description for the given prediction and actual-labels data.

        Args:
            prediction_metric_config (dict): Specification for prediction metric to run in this job.
            project_id (str): Project to use for the prediction metrics. Defaults to the project for the input feature_group, if the feature_group has exactly one project.

        Returns:
            PredictionMetric: The Prediction Metric job description.
        """
        return self.client.create_prediction_metric(self.feature_group_id, prediction_metric_config, project_id)

    def list_prediction_metrics(self, limit: int = 100, should_include_latest_version_description: bool = True, start_after_id: str = None):
        """
        List the prediction metrics for a feature group.

        Args:
            limit (int): The the number of prediction metrics to be retrieved.
            should_include_latest_version_description (bool): include the description of the latest prediction metric version for each prediction metric
            start_after_id (str): An offset parameter to exclude all prediction metrics till the specified prediction metric ID.

        Returns:
            PredictionMetric: The prediction metrics for this feature group.
        """
        return self.client.list_prediction_metrics(self.feature_group_id, limit, should_include_latest_version_description, start_after_id)

    def query_prediction_metrics(self, project_id: str = None, limit: int = 100, should_include_latest_version_description: bool = True, start_after_id: str = None):
        """
        Query and return prediction metrics and extra data needed by the UI, constrained by the parameters provided.

        feature_group_id (Unique String Identifier): [optional] The feature group used as input to the prediction metrics.
            project_id (Unique String Identifier): [optional] The project_id of the prediction metrics.
            limit (Integer): The the number of prediction metrics to be retrieved.
            should_include_latest_version_description (Boolean): include the description of the latest prediction metric version for each prediction metric
            start_after_id (Unique String Identifier): An offset parameter to exclude all prediction metrics till the specified prediction metric ID.


        Args:
            project_id (str): 
            limit (int): 
            should_include_latest_version_description (bool): 
            start_after_id (str): 

        Returns:
            PredictionMetric: The prediction metrics for this feature group.
        """
        return self.client.query_prediction_metrics(self.feature_group_id, project_id, limit, should_include_latest_version_description, start_after_id)

    def upsert_data(self, streaming_token: str, data: dict):
        """
        Updates new data into the feature group for a given lookup key recordId if the recordID is found otherwise inserts new data into the feature group.

        Args:
            streaming_token (str): The streaming token for authenticating requests
            data (dict): The data to record
        """
        return self.client.upsert_data(self.feature_group_id, streaming_token, data)

    def append_data(self, streaming_token: str, data: dict):
        """
        Appends new data into the feature group for a given lookup key recordId.

        Args:
            streaming_token (str): The streaming token for authenticating requests
            data (dict): The data to record
        """
        return self.client.append_data(self.feature_group_id, streaming_token, data)

    def upsert_multiple_data(self, streaming_token: str, data: dict):
        """
        Updates new data into the feature group for a given lookup key recordId if the recordID is found otherwise inserts new data into the feature group.

        Args:
            streaming_token (str): The streaming token for authenticating requests
            data (dict): The data to record, as an array of JSON Objects
        """
        return self.client.upsert_multiple_data(self.feature_group_id, streaming_token, data)

    def append_multiple_data(self, streaming_token: str, data: list):
        """
        Appends new data into the feature group for a given lookup key recordId.

        Args:
            streaming_token (str): The streaming token for authenticating requests
            data (list): The data to record, as an array of JSON objects
        """
        return self.client.append_multiple_data(self.feature_group_id, streaming_token, data)

    def wait_for_dataset(self, timeout: int = 7200):
        """
            A waiting call until the feature group's dataset, if any, is ready for use.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 7200 seconds.
        """
        dataset_id = self.dataset_id
        if not dataset_id:
            return  # todo check return value type
        dataset = self.client.describe_dataset(dataset_id=dataset_id)
        dataset.wait_for_inspection(timeout=timeout)
        return self.refresh()

    def wait_for_upload(self, timeout: int = 7200):
        """
            Waits for a feature group created from a dataframe to be ready for materialization and version creation.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 7200 seconds.
        """
        self.wait_for_dataset(timeout=timeout)
        return self.refresh()

    def wait_for_materialization(self, timeout: int = 7200):
        """
        A waiting call until feature group is materialized.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 7200 seconds.
        """
        latest_feature_group_version = self.describe().latest_feature_group_version
        if not latest_feature_group_version:
            from .client import ApiException
            raise ApiException(
                409, 'This feature group does not have any versions')
        self.latest_feature_group_version = latest_feature_group_version.wait_for_materialization(
            timeout=timeout)
        return self

    def wait_for_streaming_ready(self, timeout: int = 600):
        """
        Waits for the feature group indexing config to be applied for streaming

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 600 seconds.
        """
        if not self.streaming_enabled:
            from .client import ApiException
            raise ApiException(409, 'Feature group is not streaming enabled')
        return self.client._poll(self, {False}, poll_args={'streaming_status': True}, timeout=timeout)

    def get_status(self, streaming_status: bool = False):
        """
        Gets the status of the feature group.

        Returns:
            str: A string describing the status of a feature group (pending, complete, etc.).
        """
        if streaming_status:
            return self.describe().streaming_ready
        return self.describe().latest_feature_group_version.status

    def load_as_pandas(self):
        """
        Loads the feature groups into a python pandas dataframe.

        Returns:
            DataFrame: A pandas dataframe with annotations and text_snippet columns.
        """
        latest_version = self.materialize().latest_feature_group_version
        return latest_version.load_as_pandas()

    def describe_dataset(self):
        """
        Displays the dataset attached to a feature group.

        Returns:
            Dataset: A dataset object with all the relevant information about the dataset.
        """
        if self.dataset_id:
            return self.client.describe_dataset(self.dataset_id)

    def materialize(self):
        """
        Materializes the feature group's latest change at the api call time. It'll skip materialization if no change since the current latest version.

        Returns:
            FeatureGroup: A feature group object with the lastest changes materialized.
        """
        self.refresh()
        if not self.latest_feature_group_version or self.latest_version_outdated:
            self.latest_feature_group_version = self.create_version().wait_for_materialization()
        return self
