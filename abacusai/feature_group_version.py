import os
from typing import List

from .annotation_config import AnnotationConfig
from .code_source import CodeSource
from .feature import Feature
from .indexing_config import IndexingConfig
from .point_in_time_group import PointInTimeGroup
from .return_class import AbstractApiClass


class FeatureGroupVersion(AbstractApiClass):
    """
        A materialized version of a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupVersion (str): The unique identifier for this materialized version of feature group.
            featureGroupId (str): The unique identifier of the feature group this version belongs to.
            sql (str): The sql definition creating this feature group.
            sourceTables (list[str]): The source tables for this feature group.
            sourceDatasetVersions (list[str]): The dataset version ids for this feature group version.
            createdAt (str): The timestamp at which the feature group version was created.
            status (str): The current status of the feature group version.
            error (str): Relevant error if the status is FAILED.
            deployable (bool): whether feature group is deployable or not.
            cpuSize (str): Cpu size specified for the python feature group.
            memory (int): Memory in GB specified for the python feature group.
            useOriginalCsvNames (bool): If true, the feature group will use the original column names in the source dataset.
            pythonFunctionBindings (list): Config specifying variable names, types, and values to use when resolving a Python feature group.
            indexingConfigWarningMsg (str): The warning message related to indexing keys.
            materializationStartedAt (str): The timestamp at which the feature group materialization started.
            materializationCompletedAt (str): The timestamp at which the feature group materialization completed.
            columns (list[feature]): List of resolved columns.
            templateBindings (list): Template variable bindings used for resolving the template.
            features (Feature): List of features.
            pointInTimeGroups (PointInTimeGroup): List of Point In Time Groups
            codeSource (CodeSource): If a python feature group, information on the source code
            annotationConfig (AnnotationConfig): The annotations config for the feature group.
            indexingConfig (IndexingConfig): The indexing config for the feature group.
    """

    def __init__(self, client, featureGroupVersion=None, featureGroupId=None, sql=None, sourceTables=None, sourceDatasetVersions=None, createdAt=None, status=None, error=None, deployable=None, cpuSize=None, memory=None, useOriginalCsvNames=None, pythonFunctionBindings=None, indexingConfigWarningMsg=None, materializationStartedAt=None, materializationCompletedAt=None, columns=None, templateBindings=None, features={}, pointInTimeGroups={}, codeSource={}, annotationConfig={}, indexingConfig={}):
        super().__init__(client, featureGroupVersion)
        self.feature_group_version = featureGroupVersion
        self.feature_group_id = featureGroupId
        self.sql = sql
        self.source_tables = sourceTables
        self.source_dataset_versions = sourceDatasetVersions
        self.created_at = createdAt
        self.status = status
        self.error = error
        self.deployable = deployable
        self.cpu_size = cpuSize
        self.memory = memory
        self.use_original_csv_names = useOriginalCsvNames
        self.python_function_bindings = pythonFunctionBindings
        self.indexing_config_warning_msg = indexingConfigWarningMsg
        self.materialization_started_at = materializationStartedAt
        self.materialization_completed_at = materializationCompletedAt
        self.columns = columns
        self.template_bindings = templateBindings
        self.features = client._build_class(Feature, features)
        self.point_in_time_groups = client._build_class(
            PointInTimeGroup, pointInTimeGroups)
        self.code_source = client._build_class(CodeSource, codeSource)
        self.annotation_config = client._build_class(
            AnnotationConfig, annotationConfig)
        self.indexing_config = client._build_class(
            IndexingConfig, indexingConfig)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'feature_group_version': repr(self.feature_group_version), f'feature_group_id': repr(self.feature_group_id), f'sql': repr(self.sql), f'source_tables': repr(self.source_tables), f'source_dataset_versions': repr(self.source_dataset_versions), f'created_at': repr(self.created_at), f'status': repr(self.status), f'error': repr(self.error), f'deployable': repr(self.deployable), f'cpu_size': repr(self.cpu_size), f'memory': repr(self.memory), f'use_original_csv_names': repr(self.use_original_csv_names), f'python_function_bindings': repr(
            self.python_function_bindings), f'indexing_config_warning_msg': repr(self.indexing_config_warning_msg), f'materialization_started_at': repr(self.materialization_started_at), f'materialization_completed_at': repr(self.materialization_completed_at), f'columns': repr(self.columns), f'template_bindings': repr(self.template_bindings), f'features': repr(self.features), f'point_in_time_groups': repr(self.point_in_time_groups), f'code_source': repr(self.code_source), f'annotation_config': repr(self.annotation_config), f'indexing_config': repr(self.indexing_config)}
        class_name = "FeatureGroupVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'feature_group_version': self.feature_group_version, 'feature_group_id': self.feature_group_id, 'sql': self.sql, 'source_tables': self.source_tables, 'source_dataset_versions': self.source_dataset_versions, 'created_at': self.created_at, 'status': self.status, 'error': self.error, 'deployable': self.deployable, 'cpu_size': self.cpu_size, 'memory': self.memory, 'use_original_csv_names': self.use_original_csv_names, 'python_function_bindings': self.python_function_bindings, 'indexing_config_warning_msg': self.indexing_config_warning_msg,
                'materialization_started_at': self.materialization_started_at, 'materialization_completed_at': self.materialization_completed_at, 'columns': self.columns, 'template_bindings': self.template_bindings, 'features': self._get_attribute_as_dict(self.features), 'point_in_time_groups': self._get_attribute_as_dict(self.point_in_time_groups), 'code_source': self._get_attribute_as_dict(self.code_source), 'annotation_config': self._get_attribute_as_dict(self.annotation_config), 'indexing_config': self._get_attribute_as_dict(self.indexing_config)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def create_snapshot_feature_group(self, table_name: str):
        """
        Creates a Snapshot Feature Group corresponding to a specific Feature Group version.

        Args:
            table_name (str): Name for the newly created Snapshot Feature Group table. Can be up to 120 characters long and can only contain alphanumeric characters and underscores.

        Returns:
            FeatureGroup: Feature Group corresponding to the newly created Snapshot.
        """
        return self.client.create_snapshot_feature_group(self.feature_group_version, table_name)

    def export_to_file_connector(self, location: str, export_file_format: str, overwrite: bool = False):
        """
        Export Feature group to File Connector.

        Args:
            location (str): Cloud file location to export to.
            export_file_format (str): Enum string specifying the file format to export to.
            overwrite (bool): If true and a file exists at this location, this process will overwrite the file.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance.
        """
        return self.client.export_feature_group_version_to_file_connector(self.feature_group_version, location, export_file_format, overwrite)

    def export_to_database_connector(self, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None):
        """
        Export Feature group to Database Connector.

        Args:
            database_connector_id (str): Unique string identifier for the Database Connector to export to.
            object_name (str): Name of the database object to write to.
            write_mode (str): Enum string indicating whether to use INSERT or UPSERT.
            database_feature_mapping (dict): Key/value pair JSON object of "database connector column" -> "feature name" pairs.
            id_column (str): Required if write_mode is UPSERT. Indicates which database column should be used as the lookup key.
            additional_id_columns (list): For database connectors which support it, additional ID columns to use as a complex key for upserting.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance.
        """
        return self.client.export_feature_group_version_to_database_connector(self.feature_group_version, database_connector_id, object_name, write_mode, database_feature_mapping, id_column, additional_id_columns)

    def export_to_console(self, export_file_format: str):
        """
        Export Feature group to console.

        Args:
            export_file_format (str): File format to export to.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance.
        """
        return self.client.export_feature_group_version_to_console(self.feature_group_version, export_file_format)

    def delete(self):
        """
        Deletes a Feature Group Version.

        Args:
            feature_group_version (str): String identifier for the feature group version to be removed.
        """
        return self.client.delete_feature_group_version(self.feature_group_version)

    def get_materialization_logs(self, stdout: bool = False, stderr: bool = False):
        """
        Returns logs for a materialized feature group version.

        Args:
            stdout (bool): Set to True to get info logs.
            stderr (bool): Set to True to get error logs.

        Returns:
            FunctionLogs: A function logs object.
        """
        return self.client.get_materialization_logs(self.feature_group_version, stdout, stderr)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            FeatureGroupVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describe a feature group version.

        Args:
            feature_group_version (str): The unique identifier associated with the feature group version.

        Returns:
            FeatureGroupVersion: The feature group version.
        """
        return self.client.describe_feature_group_version(self.feature_group_version)

    def get_metrics(self, selected_columns: List = None, include_charts: bool = False, include_statistics: bool = True):
        """
        Get metrics for a specific feature group version.

        Args:
            selected_columns (List): A list of columns to order first.
            include_charts (bool): A flag indicating whether charts should be included in the response. Default is false.
            include_statistics (bool): A flag indicating whether statistics should be included in the response. Default is true.

        Returns:
            DataMetrics: The metrics for the specified feature group version.
        """
        return self.client.get_feature_group_version_metrics(self.feature_group_version, selected_columns, include_charts, include_statistics)

    def get_logs(self):
        """
        Retrieves the feature group materialization logs.

        Args:
            feature_group_version (str): The unique version ID of the feature group version.

        Returns:
            FeatureGroupVersionLogs: The logs for the specified feature group version.
        """
        return self.client.get_feature_group_version_logs(self.feature_group_version)

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until feature group version is materialized

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'GENERATING'}, timeout=timeout)

    def wait_for_materialization(self, timeout=3600):
        """
        A waiting call until feature group version is materialized.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_results(timeout)

    def get_status(self):
        """
        Gets the status of the feature group version.

        Returns:
            str: A string describing the status of a feature group version (pending, complete, etc.).
        """
        return self.describe().status

    # internal call
    def _download_avro_file(self, file_part, tmp_dir, part_index):
        from .api_client_utils import try_abacus_internal_copy

        part_path = os.path.join(tmp_dir, f'{part_index}.avro')
        if try_abacus_internal_copy(file_part, part_path):
            return part_path

        offset = 0
        with open(part_path, 'wb') as file:
            while True:
                with self.client._call_api('_downloadFeatureGroupVersionPartChunk', 'GET', query_params={'featureGroupVersion': self.id, 'part': part_index, 'offset': offset}, streamable_response=True, retry_500=True) as chunk:
                    bytes_written = file.write(chunk.read())
                    if not bytes_written:
                        break
                    offset += bytes_written

        return part_path

    def load_as_pandas(self, max_workers=10):
        """
        Loads the feature group version into a pandas dataframe.

        Args:
            max_workers (int): The number of threads.

        Returns:
            DataFrame: A pandas dataframe displaying the data in the feature group version.
        """

        from .api_client_utils import load_as_pandas_from_avro_files

        file_parts = self.client._call_api(
            '_getFeatureGroupVersionParts', 'GET', query_params={'featureGroupVersion': self.id}, retry_500=True)
        return load_as_pandas_from_avro_files(file_parts, self._download_avro_file, max_workers=max_workers)

    def load_as_pandas_documents(self, doc_id_column: str = 'doc_id', document_column: str = 'page_infos', max_workers=10):
        """
        Loads a feature group with documents data into a pandas dataframe.

        Args:
            doc_id_column (str): The name of the feature / column containing the document ID.
            document_column (str): The name of the feature / column which either contains the document data itself or page infos with path to remotely stored documents. This column will be replaced with the extracted document data.
            max_workers (int): The number of threads.

        Returns:
            DataFrame: A pandas dataframe containing the extracted document data.
        """

        from .api_client_utils import DocstoreUtils

        def get_docstore_resource_bytes(feature_group_version, resource_type, archive_id=None, offset=None, size=None, result_zip_path=None):
            with self.client._call_api('_downloadDocstoreResourceChunk', 'GET',
                                       query_params={'featureGroupVersion': feature_group_version, 'resourceType':
                                                     resource_type, 'archiveId': archive_id, 'offset': offset, 'size': size,
                                                     'result_zip_path': result_zip_path},
                                       streamable_response=True, retry_500=True) as chunk:
                bytes = chunk.read()
            return bytes

        def get_document_processing_result_infos(content_hash_list, document_processing_config, document_processing_version=None):
            return self.client._proxy_request('_getDocumentProcessingResultInfos', 'POST',
                                              body={'contentHashList': content_hash_list,
                                                    'documentProcessingConfig': document_processing_config,
                                                    'documentProcessingVersion': document_processing_version},
                                              is_sync=True)

        feature_group_version = self.id
        df = self.load_as_pandas(max_workers=max_workers)
        return DocstoreUtils.get_pandas_documents_df(df, feature_group_version, doc_id_column, document_column,
                                                     get_docstore_resource_bytes, get_document_processing_result_infos, max_workers=max_workers)
