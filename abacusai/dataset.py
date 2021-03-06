from .dataset_version import DatasetVersion


class Dataset():
    '''

    '''

    def __init__(self, client, datasetId=None, name=None, sourceType=None, dataSource=None, createdAt=None, refreshSchedules=None, ignoreBefore=None, ephemeral=None, lookbackDays=None, databaseConnectorId=None, databaseConnectorConfig=None, connectorType=None, datasetTableName=None, applicationConnectorId=None, applicationConnectorConfig=None, latestDatasetVersion={}):
        self.client = client
        self.id = datasetId
        self.dataset_id = datasetId
        self.name = name
        self.source_type = sourceType
        self.data_source = dataSource
        self.created_at = createdAt
        self.refresh_schedules = refreshSchedules
        self.ignore_before = ignoreBefore
        self.ephemeral = ephemeral
        self.lookback_days = lookbackDays
        self.database_connector_id = databaseConnectorId
        self.database_connector_config = databaseConnectorConfig
        self.connector_type = connectorType
        self.dataset_table_name = datasetTableName
        self.application_connector_id = applicationConnectorId
        self.application_connector_config = applicationConnectorConfig
        self.latest_dataset_version = client._build_class(
            DatasetVersion, latestDatasetVersion)

    def __repr__(self):
        return f"Dataset(dataset_id={repr(self.dataset_id)}, name={repr(self.name)}, source_type={repr(self.source_type)}, data_source={repr(self.data_source)}, created_at={repr(self.created_at)}, refresh_schedules={repr(self.refresh_schedules)}, ignore_before={repr(self.ignore_before)}, ephemeral={repr(self.ephemeral)}, lookback_days={repr(self.lookback_days)}, database_connector_id={repr(self.database_connector_id)}, database_connector_config={repr(self.database_connector_config)}, connector_type={repr(self.connector_type)}, dataset_table_name={repr(self.dataset_table_name)}, application_connector_id={repr(self.application_connector_id)}, application_connector_config={repr(self.application_connector_config)}, latest_dataset_version={repr(self.latest_dataset_version)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'dataset_id': self.dataset_id, 'name': self.name, 'source_type': self.source_type, 'data_source': self.data_source, 'created_at': self.created_at, 'refresh_schedules': self.refresh_schedules, 'ignore_before': self.ignore_before, 'ephemeral': self.ephemeral, 'lookback_days': self.lookback_days, 'database_connector_id': self.database_connector_id, 'database_connector_config': self.database_connector_config, 'connector_type': self.connector_type, 'dataset_table_name': self.dataset_table_name, 'application_connector_id': self.application_connector_id, 'application_connector_config': self.application_connector_config, 'latest_dataset_version': [elem.to_dict() for elem in self.latest_dataset_version or []]}

    def create_version_from_file_connector(self, location=None, file_format=None, csv_delimiter=None):
        return self.client.create_dataset_version_from_file_connector(self.dataset_id, location, file_format, csv_delimiter)

    def create_version_from_database_connector(self, object_name=None, columns=None, query_arguments=None, sql_query=None):
        return self.client.create_dataset_version_from_database_connector(self.dataset_id, object_name, columns, query_arguments, sql_query)

    def create_version_from_application_connector(self, object_id=None, start_date=None, end_date=None):
        return self.client.create_dataset_version_from_application_connector(self.dataset_id, object_id, start_date, end_date)

    def create_version_from_upload(self, file_format=None):
        return self.client.create_dataset_version_from_upload(self.dataset_id, file_format)

    def snapshot_streaming_data(self):
        return self.client.snapshot_streaming_data(self.dataset_id)

    def create_version(self, location=None, file_format=None, csv_delimiter=None):
        return self.client.create_dataset_version(self.dataset_id, location, file_format, csv_delimiter)

    def create_version_from_local_file(self, file_format=None):
        return self.client.create_dataset_version_from_local_file(self.dataset_id, file_format)

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_dataset(self.dataset_id)

    def list_versions(self, limit=100, start_after_version=None):
        return self.client.list_dataset_versions(self.dataset_id, limit, start_after_version)

    def attach_to_project(self, project_id, dataset_type):
        return self.client.attach_dataset_to_project(self.dataset_id, project_id, dataset_type)

    def remove_from_project(self, project_id):
        return self.client.remove_dataset_from_project(self.dataset_id, project_id)

    def rename(self, name):
        return self.client.rename_dataset(self.dataset_id, name)

    def delete(self):
        return self.client.delete_dataset(self.dataset_id)

    def wait_for_import(self, timeout=900):
        return self.client._poll(self, {'PENDING', 'IMPORTING'}, timeout=timeout)

    def wait_for_inspection(self, timeout=None):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'IMPORTING', 'CONVERTING', 'INSPECTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().latest_dataset_version.status
