from typing import List

from .return_class import AbstractApiClass


class DatasetVersion(AbstractApiClass):
    """
        A specific version of a dataset

        Args:
            client (ApiClient): An authenticated API Client instance
            datasetVersion (str): The unique identifier of the dataset version.
            status (str): The current status of the dataset version
            datasetId (str): A reference to the Dataset this dataset version belongs to.
            size (int): The size in bytes of the file.
            rowCount (int): Number of rows in the dataset version.
            fileInspectMetadata (dict): Metadata information about file's inspection. For example - the detected delimiter for CSV files.
            createdAt (str): The timestamp this dataset version was created.
            error (str): If status is FAILED, this field will be populated with an error.
            incrementalQueriedAt (str): If the dataset version is from an incremental dataset, this is the last entry of timestamp column when the dataset version was created.
            uploadId (str): If the dataset version is being uploaded, this the reference to the Upload
            mergeFileSchemas (bool): If the merge file schemas policy is enabled.
            databaseConnectorConfig (dict): The database connector query used to retrieve data for this version.
            applicationConnectorConfig (dict): The application connector used to retrieve data for this version.
            invalidRecords (str): Invalid records in the dataset version
    """

    def __init__(self, client, datasetVersion=None, status=None, datasetId=None, size=None, rowCount=None, fileInspectMetadata=None, createdAt=None, error=None, incrementalQueriedAt=None, uploadId=None, mergeFileSchemas=None, databaseConnectorConfig=None, applicationConnectorConfig=None, invalidRecords=None):
        super().__init__(client, datasetVersion)
        self.dataset_version = datasetVersion
        self.status = status
        self.dataset_id = datasetId
        self.size = size
        self.row_count = rowCount
        self.file_inspect_metadata = fileInspectMetadata
        self.created_at = createdAt
        self.error = error
        self.incremental_queried_at = incrementalQueriedAt
        self.upload_id = uploadId
        self.merge_file_schemas = mergeFileSchemas
        self.database_connector_config = databaseConnectorConfig
        self.application_connector_config = applicationConnectorConfig
        self.invalid_records = invalidRecords
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'dataset_version': repr(self.dataset_version), f'status': repr(self.status), f'dataset_id': repr(self.dataset_id), f'size': repr(self.size), f'row_count': repr(self.row_count), f'file_inspect_metadata': repr(self.file_inspect_metadata), f'created_at': repr(self.created_at), f'error': repr(self.error), f'incremental_queried_at': repr(
            self.incremental_queried_at), f'upload_id': repr(self.upload_id), f'merge_file_schemas': repr(self.merge_file_schemas), f'database_connector_config': repr(self.database_connector_config), f'application_connector_config': repr(self.application_connector_config), f'invalid_records': repr(self.invalid_records)}
        class_name = "DatasetVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'dataset_version': self.dataset_version, 'status': self.status, 'dataset_id': self.dataset_id, 'size': self.size, 'row_count': self.row_count, 'file_inspect_metadata': self.file_inspect_metadata, 'created_at': self.created_at, 'error': self.error, 'incremental_queried_at':
                self.incremental_queried_at, 'upload_id': self.upload_id, 'merge_file_schemas': self.merge_file_schemas, 'database_connector_config': self.database_connector_config, 'application_connector_config': self.application_connector_config, 'invalid_records': self.invalid_records}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def get_metrics(self, selected_columns: List = None, include_charts: bool = False, include_statistics: bool = True):
        """
        Get metrics for a specific dataset version.

        Args:
            selected_columns (List): A list of columns to order first.
            include_charts (bool): A flag indicating whether charts should be included in the response. Default is false.
            include_statistics (bool): A flag indicating whether statistics should be included in the response. Default is true.

        Returns:
            DataMetrics: The metrics for the specified Dataset version.
        """
        return self.client.get_dataset_version_metrics(self.dataset_version, selected_columns, include_charts, include_statistics)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            DatasetVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified dataset version, including its ID, name, source type, and other attributes.

        Args:
            dataset_version (str): Unique string identifier associated with the dataset version.

        Returns:
            DatasetVersion: The dataset version.
        """
        return self.client.describe_dataset_version(self.dataset_version)

    def delete(self):
        """
        Deletes the specified dataset version from the organization.

        Args:
            dataset_version (str): String identifier of the dataset version to delete.
        """
        return self.client.delete_dataset_version(self.dataset_version)

    def get_logs(self):
        """
        Retrieves the dataset import logs.

        Args:
            dataset_version (str): The unique version ID of the dataset version.

        Returns:
            DatasetVersionLogs: The logs for the specified dataset version.
        """
        return self.client.get_dataset_version_logs(self.dataset_version)

    def wait_for_import(self, timeout=900):
        """
        A waiting call until dataset version is imported.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'IMPORTING'}, timeout=timeout)

    def wait_for_inspection(self, timeout=None):
        """
        A waiting call until dataset version is completely inspected.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.

        """
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'IMPORTING', 'CONVERTING', 'INSPECTING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the dataset version.

        Returns:
            str: A string describing the status of a dataset version (importing, inspecting, complete, etc.).
        """
        return self.describe().status
