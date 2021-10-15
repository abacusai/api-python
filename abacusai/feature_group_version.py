import time
import io
from .feature import Feature
from concurrent.futures import ThreadPoolExecutor


class FeatureGroupVersion():
    '''
        A materialized version of a feature group
    '''

    def __init__(self, client, featureGroupVersion=None, sql=None, sourceTables=None, createdAt=None, status=None, error=None, deployable=None, features={}):
        self.client = client
        self.id = featureGroupVersion
        self.feature_group_version = featureGroupVersion
        self.sql = sql
        self.source_tables = sourceTables
        self.created_at = createdAt
        self.status = status
        self.error = error
        self.deployable = deployable
        self.features = client._build_class(Feature, features)

    def __repr__(self):
        return f"FeatureGroupVersion(feature_group_version={repr(self.feature_group_version)}, sql={repr(self.sql)}, source_tables={repr(self.source_tables)}, created_at={repr(self.created_at)}, status={repr(self.status)}, error={repr(self.error)}, deployable={repr(self.deployable)}, features={repr(self.features)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_version': self.feature_group_version, 'sql': self.sql, 'source_tables': self.source_tables, 'created_at': self.created_at, 'status': self.status, 'error': self.error, 'deployable': self.deployable, 'features': self.features.to_dict() if self.features else None}

    def export_to_file_connector(self, location, export_file_format):
        return self.client.export_feature_group_version_to_file_connector(self.feature_group_version, location, export_file_format)

    def export_to_database_connector(self, database_connector_id, object_name, write_mode, database_feature_mapping, id_column=None):
        return self.client.export_feature_group_version_to_database_connector(self.feature_group_version, database_connector_id, object_name, write_mode, database_feature_mapping, id_column)

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_feature_group_version(self.feature_group_version)

    def wait_for_results(self, timeout=3600):
        return self.client._poll(self, {'PENDING'}, timeout=timeout)

    def get_status(self):
        return self.client._call_api('describeFeatureGroupVersion', 'GET', query_params={'featureGroupVersion': self.feature_group_version}, parse_type=FeatureGroupVersion).status

    def describe(self):
        return self.client._call_api('describeFeatureGroupVersion', 'GET', query_params={'featureGroupVersion': self.feature_group_version}, parse_type=FeatureGroupVersion)

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
        import pandas as pd
        import fastavro

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
