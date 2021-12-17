import io
from concurrent.futures import ThreadPoolExecutor

from .feature import Feature
from .return_class import AbstractApiClass


class FeatureGroupVersion(AbstractApiClass):
    """
        A materialized version of a feature group
    """

    def __init__(self, client, featureGroupVersion=None, sql=None, sourceTables=None, createdAt=None, status=None, error=None, deployable=None, features={}):
        super().__init__(client, featureGroupVersion)
        self.feature_group_version = featureGroupVersion
        self.sql = sql
        self.source_tables = sourceTables
        self.created_at = createdAt
        self.status = status
        self.error = error
        self.deployable = deployable
        self.features = client._build_class(Feature, features)

    def __repr__(self):
        return f"FeatureGroupVersion(feature_group_version={repr(self.feature_group_version)},\n  sql={repr(self.sql)},\n  source_tables={repr(self.source_tables)},\n  created_at={repr(self.created_at)},\n  status={repr(self.status)},\n  error={repr(self.error)},\n  deployable={repr(self.deployable)},\n  features={repr(self.features)})"

    def to_dict(self):
        return {'feature_group_version': self.feature_group_version, 'sql': self.sql, 'source_tables': self.source_tables, 'created_at': self.created_at, 'status': self.status, 'error': self.error, 'deployable': self.deployable, 'features': self._get_attribute_as_dict(self.features)}

    def export_to_file_connector(self, location, export_file_format, overwrite=False):
        """Export Feature group to File Connector."""
        return self.client.export_feature_group_version_to_file_connector(self.feature_group_version, location, export_file_format, overwrite)

    def export_to_database_connector(self, database_connector_id, object_name, write_mode, database_feature_mapping, id_column=None):
        """Export Feature group to Database Connector."""
        return self.client.export_feature_group_version_to_database_connector(self.feature_group_version, database_connector_id, object_name, write_mode, database_feature_mapping, id_column)

    def get_materialization_logs(self, stdout=False, stderr=False):
        """Returns logs for materialized feature group version."""
        return self.client.get_materialization_logs(self.feature_group_version, stdout, stderr)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Get a specific feature group version."""
        return self.client.describe_feature_group_version(self.feature_group_version)

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until feature group version is created.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 3600 milliseconds.

        Returns:
            None
        """
        return self.client._poll(self, {'PENDING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the feature group version.

        Returns:
            Enum (string): A string describing the status of a feature group version (pending, complete, etc.).
        """
        return self.describe().status

    # internal call
    def _get_avro_file(self, file_part):
        file = io.BytesIO()
        offset = 0
        while True:
            with self.client._call_api('_downloadFeatureGroupVersionPartChunk', 'GET', query_params={'featureGroupVersion': self.id, 'part': file_part, 'offset': offset}, streamable_response=True) as chunk:
                bytes_written = file.write(chunk.read())
            if not bytes_written:
                break
            offset += bytes_written
        file.seek(0)
        return file

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
            file_futures = [executor.submit(
                self._get_avro_file, file_part) for file_part in file_parts]
            for future in file_futures:
                data = future.result()
                reader = fastavro.reader(data)
                data_df = data_df.append(
                    pd.DataFrame.from_records([r for r in reader]))
        return data_df
