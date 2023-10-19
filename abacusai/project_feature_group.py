from .annotation_config import AnnotationConfig
from .code_source import CodeSource
from .concatenation_config import ConcatenationConfig
from .feature import Feature
from .feature_group import FeatureGroup
from .feature_group_template import FeatureGroupTemplate
from .feature_group_version import FeatureGroupVersion
from .indexing_config import IndexingConfig
from .natural_language_explanation import NaturalLanguageExplanation
from .point_in_time_group import PointInTimeGroup
from .project_config import ProjectConfig
from .project_feature_group_schema import ProjectFeatureGroupSchema
from .refresh_schedule import RefreshSchedule


class ProjectFeatureGroup(FeatureGroup):
    """
        A feature group along with project specific mappings

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupId (str): Unique identifier for this feature group.
            modificationLock (bool): If feature group is locked against a change or not.
            name (str): [DEPRECATED] User friendly name for the feature group.
            featureGroupSourceType (str): The source type of the feature group
            tableName (str): Unique table name of this feature group.
            sql (str): SQL definition creating this feature group.
            datasetId (str): Dataset ID the feature group is sourced from.
            functionSourceCode (str): Source definition creating this feature group.
            functionName (str): Function name to execute from the source code.
            sourceTables (list[str]): Source tables for this feature group.
            createdAt (str): Timestamp at which the feature group was created.
            description (str): Description of the feature group.
            sqlError (str): Error message with this feature group.
            latestVersionOutdated (bool): Is latest materialized feature group version outdated.
            referencedFeatureGroups (list[str]): Feature groups this feature group is used in.
            tags (list[str]): Tags added to this feature group.
            primaryKey (str): Primary index feature.
            updateTimestampKey (str): Primary timestamp feature.
            lookupKeys (list[str]): Additional indexed features for this feature group.
            streamingEnabled (bool): If true, the feature group can have data streamed to it.
            incremental (bool): If feature group corresponds to an incremental dataset.
            mergeConfig (dict): Merge configuration settings for the feature group.
            transformConfig (dict): Transform configuration settings for the feature group.
            samplingConfig (dict): Sampling configuration for the feature group.
            cpuSize (str): CPU size specified for the Python feature group.
            memory (int): Memory in GB specified for the Python feature group.
            streamingReady (bool): If true, the feature group is ready to receive streaming data.
            featureTags (dict): Tags for features in this feature group
            moduleName (str): Path to the file with the feature group function.
            templateBindings (dict): Config specifying variable names and values to use when resolving a feature group template.
            featureExpression (str): If the dataset feature group has custom features, the SQL select expression creating those features.
            useOriginalCsvNames (bool): If true, the feature group will use the original column names in the source dataset.
            pythonFunctionBindings (dict): Config specifying variable names, types, and values to use when resolving a Python feature group.
            pythonFunctionName (str): Name of the Python function the feature group was built from.
            useGpu (bool): Whether this feature group is using gpu
            featureGroupType (str): Project type when the feature group is used in the context of a project.
            features (Feature): List of resolved features.
            duplicateFeatures (Feature): List of duplicate features.
            pointInTimeGroups (PointInTimeGroup): List of Point In Time Groups.
            annotationConfig (AnnotationConfig): Annotation config for this feature
            latestFeatureGroupVersion (FeatureGroupVersion): Latest feature group version.
            concatenationConfig (ConcatenationConfig): Feature group ID whose data will be concatenated into this feature group.
            indexingConfig (IndexingConfig): Indexing config for the feature group for feature store
            codeSource (CodeSource): If a Python feature group, information on the source code.
            featureGroupTemplate (FeatureGroupTemplate): FeatureGroupTemplate to use when this feature group is attached to a template.
            explanation (NaturalLanguageExplanation): Natural language explanation of the feature group
            refreshSchedules (RefreshSchedule): List of schedules that determines when the next version of the feature group will be created.
            projectFeatureGroupSchema (ProjectFeatureGroupSchema): Project-specific schema for this feature group.
            projectConfig (ProjectConfig): Project-specific config for this feature group.
    """

    def __init__(self, client, featureGroupId=None, modificationLock=None, name=None, featureGroupSourceType=None, tableName=None, sql=None, datasetId=None, functionSourceCode=None, functionName=None, sourceTables=None, createdAt=None, description=None, sqlError=None, latestVersionOutdated=None, referencedFeatureGroups=None, tags=None, primaryKey=None, updateTimestampKey=None, lookupKeys=None, streamingEnabled=None, incremental=None, mergeConfig=None, transformConfig=None, samplingConfig=None, cpuSize=None, memory=None, streamingReady=None, featureTags=None, moduleName=None, templateBindings=None, featureExpression=None, useOriginalCsvNames=None, pythonFunctionBindings=None, pythonFunctionName=None, useGpu=None, featureGroupType=None, features={}, duplicateFeatures={}, pointInTimeGroups={}, annotationConfig={}, concatenationConfig={}, indexingConfig={}, codeSource={}, featureGroupTemplate={}, explanation={}, refreshSchedules={}, projectFeatureGroupSchema={}, projectConfig={}, latestFeatureGroupVersion={}):
        super().__init__(client, featureGroupId)
        self.feature_group_id = featureGroupId
        self.modification_lock = modificationLock
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
        self.sql_error = sqlError
        self.latest_version_outdated = latestVersionOutdated
        self.referenced_feature_groups = referencedFeatureGroups
        self.tags = tags
        self.primary_key = primaryKey
        self.update_timestamp_key = updateTimestampKey
        self.lookup_keys = lookupKeys
        self.streaming_enabled = streamingEnabled
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
        self.use_gpu = useGpu
        self.feature_group_type = featureGroupType
        self.features = client._build_class(Feature, features)
        self.duplicate_features = client._build_class(
            Feature, duplicateFeatures)
        self.point_in_time_groups = client._build_class(
            PointInTimeGroup, pointInTimeGroups)
        self.annotation_config = client._build_class(
            AnnotationConfig, annotationConfig)
        self.concatenation_config = client._build_class(
            ConcatenationConfig, concatenationConfig)
        self.indexing_config = client._build_class(
            IndexingConfig, indexingConfig)
        self.code_source = client._build_class(CodeSource, codeSource)
        self.feature_group_template = client._build_class(
            FeatureGroupTemplate, featureGroupTemplate)
        self.explanation = client._build_class(
            NaturalLanguageExplanation, explanation)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.project_feature_group_schema = client._build_class(
            ProjectFeatureGroupSchema, projectFeatureGroupSchema)
        self.project_config = client._build_class(ProjectConfig, projectConfig)
        self.latest_feature_group_version = client._build_class(
            FeatureGroupVersion, latestFeatureGroupVersion)

    def __repr__(self):
        repr_dict = {f'feature_group_id': repr(self.feature_group_id), f'modification_lock': repr(self.modification_lock), f'name': repr(self.name), f'feature_group_source_type': repr(self.feature_group_source_type), f'table_name': repr(self.table_name), f'sql': repr(self.sql), f'dataset_id': repr(self.dataset_id), f'function_source_code': repr(self.function_source_code), f'function_name': repr(self.function_name), f'source_tables': repr(self.source_tables), f'created_at': repr(self.created_at), f'description': repr(self.description), f'sql_error': repr(self.sql_error), f'latest_version_outdated': repr(self.latest_version_outdated), f'referenced_feature_groups': repr(self.referenced_feature_groups), f'tags': repr(self.tags), f'primary_key': repr(self.primary_key), f'update_timestamp_key': repr(self.update_timestamp_key), f'lookup_keys': repr(self.lookup_keys), f'streaming_enabled': repr(self.streaming_enabled), f'incremental': repr(self.incremental), f'merge_config': repr(self.merge_config), f'transform_config': repr(self.transform_config), f'sampling_config': repr(self.sampling_config), f'cpu_size': repr(self.cpu_size), f'memory': repr(self.memory), f'streaming_ready': repr(
            self.streaming_ready), f'feature_tags': repr(self.feature_tags), f'module_name': repr(self.module_name), f'template_bindings': repr(self.template_bindings), f'feature_expression': repr(self.feature_expression), f'use_original_csv_names': repr(self.use_original_csv_names), f'python_function_bindings': repr(self.python_function_bindings), f'python_function_name': repr(self.python_function_name), f'use_gpu': repr(self.use_gpu), f'feature_group_type': repr(self.feature_group_type), f'features': repr(self.features), f'duplicate_features': repr(self.duplicate_features), f'point_in_time_groups': repr(self.point_in_time_groups), f'annotation_config': repr(self.annotation_config), f'concatenation_config': repr(self.concatenation_config), f'indexing_config': repr(self.indexing_config), f'code_source': repr(self.code_source), f'feature_group_template': repr(self.feature_group_template), f'explanation': repr(self.explanation), f'refresh_schedules': repr(self.refresh_schedules), f'project_feature_group_schema': repr(self.project_feature_group_schema), f'project_config': repr(self.project_config), f'latest_feature_group_version': repr(self.latest_feature_group_version)}
        class_name = "ProjectFeatureGroup"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_group_id': self.feature_group_id, 'modification_lock': self.modification_lock, 'name': self.name, 'feature_group_source_type': self.feature_group_source_type, 'table_name': self.table_name, 'sql': self.sql, 'dataset_id': self.dataset_id, 'function_source_code': self.function_source_code, 'function_name': self.function_name, 'source_tables': self.source_tables, 'created_at': self.created_at, 'description': self.description, 'sql_error': self.sql_error, 'latest_version_outdated': self.latest_version_outdated, 'referenced_feature_groups': self.referenced_feature_groups, 'tags': self.tags, 'primary_key': self.primary_key, 'update_timestamp_key': self.update_timestamp_key, 'lookup_keys': self.lookup_keys, 'streaming_enabled': self.streaming_enabled, 'incremental': self.incremental, 'merge_config': self.merge_config, 'transform_config': self.transform_config, 'sampling_config': self.sampling_config, 'cpu_size': self.cpu_size, 'memory': self.memory, 'streaming_ready': self.streaming_ready, 'feature_tags': self.feature_tags, 'module_name': self.module_name, 'template_bindings': self.template_bindings, 'feature_expression': self.feature_expression,
                'use_original_csv_names': self.use_original_csv_names, 'python_function_bindings': self.python_function_bindings, 'python_function_name': self.python_function_name, 'use_gpu': self.use_gpu, 'feature_group_type': self.feature_group_type, 'features': self._get_attribute_as_dict(self.features), 'duplicate_features': self._get_attribute_as_dict(self.duplicate_features), 'point_in_time_groups': self._get_attribute_as_dict(self.point_in_time_groups), 'annotation_config': self._get_attribute_as_dict(self.annotation_config), 'concatenation_config': self._get_attribute_as_dict(self.concatenation_config), 'indexing_config': self._get_attribute_as_dict(self.indexing_config), 'code_source': self._get_attribute_as_dict(self.code_source), 'feature_group_template': self._get_attribute_as_dict(self.feature_group_template), 'explanation': self._get_attribute_as_dict(self.explanation), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'project_feature_group_schema': self._get_attribute_as_dict(self.project_feature_group_schema), 'project_config': self._get_attribute_as_dict(self.project_config), 'latest_feature_group_version': self._get_attribute_as_dict(self.latest_feature_group_version)}
        return {key: value for key, value in resp.items() if value is not None}
