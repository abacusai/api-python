from .dataset_column import DatasetColumn
from .dataset_version import DatasetVersion
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class Dataset(AbstractApiClass):
    """
        A dataset reference
    """

    def __init__(self, client, datasetId=None, name=None, sourceType=None, dataSource=None, createdAt=None, ignoreBefore=None, ephemeral=None, lookbackDays=None, databaseConnectorId=None, databaseConnectorConfig=None, connectorType=None, featureGroupTableName=None, applicationConnectorId=None, applicationConnectorConfig=None, schema={}, refreshSchedules={}, latestDatasetVersion={}):
        super().__init__(client, datasetId)
        self.dataset_id = datasetId
        self.name = name
        self.source_type = sourceType
        self.data_source = dataSource
        self.created_at = createdAt
        self.ignore_before = ignoreBefore
        self.ephemeral = ephemeral
        self.lookback_days = lookbackDays
        self.database_connector_id = databaseConnectorId
        self.database_connector_config = databaseConnectorConfig
        self.connector_type = connectorType
        self.feature_group_table_name = featureGroupTableName
        self.application_connector_id = applicationConnectorId
        self.application_connector_config = applicationConnectorConfig
        self.schema = client._build_class(DatasetColumn, schema)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.latest_dataset_version = client._build_class(
            DatasetVersion, latestDatasetVersion)

    def __repr__(self):
        return f"Dataset(dataset_id={repr(self.dataset_id)},\n  name={repr(self.name)},\n  source_type={repr(self.source_type)},\n  data_source={repr(self.data_source)},\n  created_at={repr(self.created_at)},\n  ignore_before={repr(self.ignore_before)},\n  ephemeral={repr(self.ephemeral)},\n  lookback_days={repr(self.lookback_days)},\n  database_connector_id={repr(self.database_connector_id)},\n  database_connector_config={repr(self.database_connector_config)},\n  connector_type={repr(self.connector_type)},\n  feature_group_table_name={repr(self.feature_group_table_name)},\n  application_connector_id={repr(self.application_connector_id)},\n  application_connector_config={repr(self.application_connector_config)},\n  schema={repr(self.schema)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  latest_dataset_version={repr(self.latest_dataset_version)})"

    def to_dict(self):
        return {'dataset_id': self.dataset_id, 'name': self.name, 'source_type': self.source_type, 'data_source': self.data_source, 'created_at': self.created_at, 'ignore_before': self.ignore_before, 'ephemeral': self.ephemeral, 'lookback_days': self.lookback_days, 'database_connector_id': self.database_connector_id, 'database_connector_config': self.database_connector_config, 'connector_type': self.connector_type, 'feature_group_table_name': self.feature_group_table_name, 'application_connector_id': self.application_connector_id, 'application_connector_config': self.application_connector_config, 'schema': self._get_attribute_as_dict(self.schema), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'latest_dataset_version': self._get_attribute_as_dict(self.latest_dataset_version)}

    def create_version_from_file_connector(self, location=None, file_format=None, csv_delimiter=None):
        """Creates a new version of the specified dataset."""
        return self.client.create_dataset_version_from_file_connector(self.dataset_id, location, file_format, csv_delimiter)

    def create_version_from_database_connector(self, object_name=None, columns=None, query_arguments=None, sql_query=None):
        """Creates a new version of the specified dataset"""
        return self.client.create_dataset_version_from_database_connector(self.dataset_id, object_name, columns, query_arguments, sql_query)

    def create_version_from_application_connector(self, object_id=None, start_timestamp=None, end_timestamp=None):
        """Creates a new version of the specified dataset"""
        return self.client.create_dataset_version_from_application_connector(self.dataset_id, object_id, start_timestamp, end_timestamp)

    def create_version_from_upload(self, file_format=None):
        """Creates a new version of the specified dataset using a local file upload."""
        return self.client.create_dataset_version_from_upload(self.dataset_id, file_format)

    def snapshot_streaming_data(self):
        """Snapshots the current data in the streaming dataset for training."""
        return self.client.snapshot_streaming_data(self.dataset_id)

    def set_column_data_type(self, column, data_type):
        """Set a column's type in a specified dataset."""
        return self.client.set_dataset_column_data_type(self.dataset_id, column, data_type)

    def set_streaming_retention_policy(self, retention_hours=None, retention_row_count=None):
        """Sets the streaming retention policy"""
        return self.client.set_streaming_retention_policy(self.dataset_id, retention_hours, retention_row_count)

    def get_schema(self):
        """Retrieves the column schema of a dataset"""
        return self.client.get_dataset_schema(self.dataset_id)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Retrieves a full description of the specified dataset, with attributes such as its ID, name, source type, etc."""
        return self.client.describe_dataset(self.dataset_id)

    def list_versions(self, limit=100, start_after_version=None):
        """Retrieves a list of all dataset versions for the specified dataset."""
        return self.client.list_dataset_versions(self.dataset_id, limit, start_after_version)

    def attach_to_project(self, project_id, dataset_type):
        """Attaches the dataset to the project."""
        return self.client.attach_dataset_to_project(self.dataset_id, project_id, dataset_type)

    def remove_from_project(self, project_id):
        """Removes a dataset from a project."""
        return self.client.remove_dataset_from_project(self.dataset_id, project_id)

    def rename(self, name):
        """Rename a dataset."""
        return self.client.rename_dataset(self.dataset_id, name)

    def delete(self):
        """Deletes the specified dataset from the organization."""
        return self.client.delete_dataset(self.dataset_id)

    def wait_for_import(self, timeout=900):
        """
        A waiting call until dataset is imported.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 900 milliseconds.

        Returns:
            None
        """
        return self.client._poll(self, {'PENDING', 'IMPORTING'}, timeout=timeout)

    def wait_for_inspection(self, timeout=None):
        """
        A waiting call until dataset is completely inspected.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.

        Returns:
            None
        """
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'IMPORTING', 'CONVERTING', 'INSPECTING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the latest dataset version.

        Returns:
            Enum (string): A string describing the status of a dataset (importing, inspecting, complete, etc.).
        """
        return self.describe().latest_dataset_version.status

    def describe_feature_group(self):
        """
        Gets the feature group attached to the dataset.

        Returns:
            FeatureGroup (object): A feature group object.
        """
        return self.client.describe_feature_group_by_table_name(self.feature_group_table_name)

    def create_refresh_policy(self, cron: str):
        """
        To create a refresh policy for a dataset.

        Args:
            cron (str): A cron style string to set the refresh time.

        Returns:
            RefreshPolicy (object): The refresh policy object.
        """
        return self.client.create_refresh_policy(self.name, cron, 'DATASET', dataset_ids=[self.id])

    def list_refresh_policies(self):
        """
        Gets the refresh policies in a list.

        Returns:
            List (RefreshPolicy): A list of refresh policy objects.
        """
        return self.client.list_refresh_policies(dataset_ids=[self.id])
