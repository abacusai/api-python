import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .code_source import CodeSource
from .feature import Feature
from .point_in_time_group import PointInTimeGroup
from .return_class import AbstractApiClass


class FeatureGroupVersion(AbstractApiClass):
    """
        A materialized version of a feature group

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupVersion (str): The unique identifier for this version of feature group.
            featureGroupId (str): 
            sql (str): The sql definition creating this feature group.
            sourceTables (list of string): The source tables for this feature group.
            createdAt (str): The timestamp at which the feature group was created.
            status (str): The current status of the feature group version.
            error (str): Relevant error if the status is FAILED.
            deployable (bool): whether feature group is deployable or not.
            cpuSize (str): Cpu size specified for the python feature group.
            memory (int): Memory in GB specified for the python feature group.
            useOriginalCsvNames (bool): If true, the feature group will use the original column names in the source dataset.
            features (Feature): List of features.
            pointInTimeGroups (PointInTimeGroup): List of Point In Time Groups
            codeSource (CodeSource): If a python feature group, information on the source code
    """

    def __init__(self, client, featureGroupVersion=None, featureGroupId=None, sql=None, sourceTables=None, createdAt=None, status=None, error=None, deployable=None, cpuSize=None, memory=None, useOriginalCsvNames=None, features={}, pointInTimeGroups={}, codeSource={}):
        super().__init__(client, featureGroupVersion)
        self.feature_group_version = featureGroupVersion
        self.feature_group_id = featureGroupId
        self.sql = sql
        self.source_tables = sourceTables
        self.created_at = createdAt
        self.status = status
        self.error = error
        self.deployable = deployable
        self.cpu_size = cpuSize
        self.memory = memory
        self.use_original_csv_names = useOriginalCsvNames
        self.features = client._build_class(Feature, features)
        self.point_in_time_groups = client._build_class(
            PointInTimeGroup, pointInTimeGroups)
        self.code_source = client._build_class(CodeSource, codeSource)

    def __repr__(self):
        return f"FeatureGroupVersion(feature_group_version={repr(self.feature_group_version)},\n  feature_group_id={repr(self.feature_group_id)},\n  sql={repr(self.sql)},\n  source_tables={repr(self.source_tables)},\n  created_at={repr(self.created_at)},\n  status={repr(self.status)},\n  error={repr(self.error)},\n  deployable={repr(self.deployable)},\n  cpu_size={repr(self.cpu_size)},\n  memory={repr(self.memory)},\n  use_original_csv_names={repr(self.use_original_csv_names)},\n  features={repr(self.features)},\n  point_in_time_groups={repr(self.point_in_time_groups)},\n  code_source={repr(self.code_source)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_version': self.feature_group_version, 'feature_group_id': self.feature_group_id, 'sql': self.sql, 'source_tables': self.source_tables, 'created_at': self.created_at, 'status': self.status, 'error': self.error, 'deployable': self.deployable, 'cpu_size': self.cpu_size, 'memory': self.memory, 'use_original_csv_names': self.use_original_csv_names, 'features': self._get_attribute_as_dict(self.features), 'point_in_time_groups': self._get_attribute_as_dict(self.point_in_time_groups), 'code_source': self._get_attribute_as_dict(self.code_source)}

    def create_snapshot_feature_group(self, table_name: str):
        """
        Creates a Snapshot Feature Group corresponding to a specific feature group version.

        Args:
            table_name (str): The name for the newly created Snapshot Feature Group table.

        Returns:
            FeatureGroup: Feature Group corresponding to the newly created Snapshot.
        """
        return self.client.create_snapshot_feature_group(self.feature_group_version, table_name)

    def export_to_file_connector(self, location: str, export_file_format: str, overwrite: bool = False):
        """
        Export Feature group to File Connector.

        Args:
            location (str): Cloud file location to export to.
            export_file_format (str): File format to export to.
            overwrite (bool): If true and a file exists at this location, this process will overwrite the file.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance
        """
        return self.client.export_feature_group_version_to_file_connector(self.feature_group_version, location, export_file_format, overwrite)

    def export_to_database_connector(self, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None):
        """
        Export Feature group to Database Connector.

        Args:
            database_connector_id (str): Database connector to export to.
            object_name (str): The database object to write to
            write_mode (str): Either INSERT or UPSERT
            database_feature_mapping (dict): A key/value pair JSON Object of "database connector column" -> "feature name" pairs.
            id_column (str): Required if mode is UPSERT. Indicates which database column should be used as the lookup key for UPSERT
            additional_id_columns (list): For database connectors which support it, additional ID columns to use as a complex key for upserting

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance
        """
        return self.client.export_feature_group_version_to_database_connector(self.feature_group_version, database_connector_id, object_name, write_mode, database_feature_mapping, id_column, additional_id_columns)

    def export_to_console(self, export_file_format: str):
        """
        Export Feature group to console.

        Args:
            export_file_format (str): File format to export to.

        Returns:
            FeatureGroupExport: The FeatureGroupExport instance
        """
        return self.client.export_feature_group_version_to_console(self.feature_group_version, export_file_format)

    def get_materialization_logs(self, stdout: bool = False, stderr: bool = False):
        """
        Returns logs for materialized feature group version.

        Args:
            stdout (bool):  Set True to get info logs
            stderr (bool):  Set True to get error logs

        Returns:
            FunctionLogs: A function logs.
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
        Get a specific feature group version.

        Args:
            feature_group_version (str): The unique ID associated with the feature group version.

        Returns:
            FeatureGroupVersion: A feature group version.
        """
        return self.client.describe_feature_group_version(self.feature_group_version)

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until feature group version is materialized

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'GENERATING'}, timeout=timeout)

    def wait_for_materialization(self, timeout=3600):
        """
        A waiting call until feature group version is materialized.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
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
    def _download_avro_file(self, file_part, tmp_dir):
        offset = 0
        part_path = os.path.join(tmp_dir, f'{file_part}.avro')
        with open(part_path, 'wb') as file:
            while True:
                with self.client._call_api('_downloadFeatureGroupVersionPartChunk', 'GET', query_params={'featureGroupVersion': self.id, 'part': file_part, 'offset': offset}, streamable_response=True) as chunk:
                    bytes_written = file.write(chunk.read())
                if not bytes_written:
                    break
                offset += bytes_written
        return part_path

    def load_as_pandas(self, max_workers=10):
        """
        Loads the feature group version into a pandas dataframe.

        Args:
            max_workers (int, optional): The number of threads.

        Returns:
            DataFrame: A pandas dataframe displaying the data in the feature group version.
        """
        import fastavro
        import pandas as pd

        file_parts = range(self.client._call_api(
            '_getFeatureGroupVersionPartCount', 'GET', query_params={'featureGroupVersion': self.id}))
        data_df = pd.DataFrame()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_futures = [executor.submit(
                    self._download_avro_file, file_part, tmp_dir) for file_part in file_parts]
                for future in file_futures:
                    part_path = future.result()
                    with open(part_path, 'rb') as part_data:
                        reader = fastavro.reader(part_data)
                        data_df = pd.concat([data_df, pd.DataFrame(
                            [r for r in reader])], ignore_index=True)
            for col in data_df.columns:
                if pd.core.dtypes.common.is_datetime64_ns_dtype(data_df[col]):
                    data_df[col] = data_df[col].dt.tz_localize(None)
        return data_df
