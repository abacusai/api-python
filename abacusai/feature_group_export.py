from .project_config import ProjectConfig
from .return_class import AbstractApiClass


class FeatureGroupExport(AbstractApiClass):
    """
        A feature Group Export Job

        Args:
            client (ApiClient): An authenticated API Client instance
            featureGroupExportId (str): Unique identifier for this export.
            failedWrites (int): Number of failed writes.
            totalWrites (int): Total number of writes.
            featureGroupVersion (str): Version of the feature group being exported.
            connectorType (str): The type of connector
            outputLocation (str): File Connector location the feature group is being written to.
            fileFormat (str): File format being written to `output_location`.
            databaseConnectorId (str): Database connector ID used.
            objectName (str): Database connector's object to write to.
            writeMode (str): `UPSERT` or `INSERT` for writing to the database connector.
            databaseFeatureMapping (dict): Column/feature pairs mapping the features to the database columns.
            idColumn (str): ID column to use as the upsert key.
            status (str): Current status of the export.
            createdAt (str): Timestamp at which the export was created (ISO-8601 format).
            exportCompletedAt (str): Timestamp at which the export completed (ISO-8601 format).
            additionalIdColumns (list[str]): For database connectors which support it, additional ID columns to use as a complex key for upserting.
            error (str): If `status` is `FAILED`, this field will be populated with an error.
            databaseOutputError (bool): If `True`, there were errors reported by the database connector while writing.
            projectConfig (ProjectConfig): Project config for this feature group.
    """

    def __init__(self, client, featureGroupExportId=None, failedWrites=None, totalWrites=None, featureGroupVersion=None, connectorType=None, outputLocation=None, fileFormat=None, databaseConnectorId=None, objectName=None, writeMode=None, databaseFeatureMapping=None, idColumn=None, status=None, createdAt=None, exportCompletedAt=None, additionalIdColumns=None, error=None, databaseOutputError=None, projectConfig={}):
        super().__init__(client, featureGroupExportId)
        self.feature_group_export_id = featureGroupExportId
        self.failed_writes = failedWrites
        self.total_writes = totalWrites
        self.feature_group_version = featureGroupVersion
        self.connector_type = connectorType
        self.output_location = outputLocation
        self.file_format = fileFormat
        self.database_connector_id = databaseConnectorId
        self.object_name = objectName
        self.write_mode = writeMode
        self.database_feature_mapping = databaseFeatureMapping
        self.id_column = idColumn
        self.status = status
        self.created_at = createdAt
        self.export_completed_at = exportCompletedAt
        self.additional_id_columns = additionalIdColumns
        self.error = error
        self.database_output_error = databaseOutputError
        self.project_config = client._build_class(ProjectConfig, projectConfig)

    def __repr__(self):
        return f"FeatureGroupExport(feature_group_export_id={repr(self.feature_group_export_id)},\n  failed_writes={repr(self.failed_writes)},\n  total_writes={repr(self.total_writes)},\n  feature_group_version={repr(self.feature_group_version)},\n  connector_type={repr(self.connector_type)},\n  output_location={repr(self.output_location)},\n  file_format={repr(self.file_format)},\n  database_connector_id={repr(self.database_connector_id)},\n  object_name={repr(self.object_name)},\n  write_mode={repr(self.write_mode)},\n  database_feature_mapping={repr(self.database_feature_mapping)},\n  id_column={repr(self.id_column)},\n  status={repr(self.status)},\n  created_at={repr(self.created_at)},\n  export_completed_at={repr(self.export_completed_at)},\n  additional_id_columns={repr(self.additional_id_columns)},\n  error={repr(self.error)},\n  database_output_error={repr(self.database_output_error)},\n  project_config={repr(self.project_config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'feature_group_export_id': self.feature_group_export_id, 'failed_writes': self.failed_writes, 'total_writes': self.total_writes, 'feature_group_version': self.feature_group_version, 'connector_type': self.connector_type, 'output_location': self.output_location, 'file_format': self.file_format, 'database_connector_id': self.database_connector_id, 'object_name': self.object_name, 'write_mode': self.write_mode, 'database_feature_mapping': self.database_feature_mapping, 'id_column': self.id_column, 'status': self.status, 'created_at': self.created_at, 'export_completed_at': self.export_completed_at, 'additional_id_columns': self.additional_id_columns, 'error': self.error, 'database_output_error': self.database_output_error, 'project_config': self._get_attribute_as_dict(self.project_config)}

    def get_feature_group_version_export_download_url(self):
        """
        Get a link to download the feature group version.

        Args:
            feature_group_export_id (str): Unique identifier of the Feature Group Export to get a signed URL for.

        Returns:
            FeatureGroupExportDownloadUrl: Instance containing the download URL and expiration time for the Feature Group Export.
        """
        return self.client.get_feature_group_version_export_download_url(self.feature_group_export_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            FeatureGroupExport: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        A feature group export

        Args:
            feature_group_export_id (str): Unique identifier of the feature group export.

        Returns:
            FeatureGroupExport: The feature group export object.
        """
        return self.client.describe_feature_group_export(self.feature_group_export_id)

    def get_connector_errors(self):
        """
        Returns a stream containing the write errors of the feature group export database connection, if any writes failed to the database connector.

        Args:
            feature_group_export_id (str): Unique identifier of the feature group export to get the errors for.
        """
        return self.client.get_feature_group_export_connector_errors(self.feature_group_export_id)

    def wait_for_results(self, timeout=3600):
        """
        A waiting call until feature group export is created.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'EXPORTING'}, timeout=timeout)

    def wait_for_export(self, timeout=3600):
        """
        A waiting call until feature group export is created.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.wait_for_results(timeout=timeout)

    def get_status(self):
        """
        Gets the status of the feature group export.

        Returns:
            str: A string describing the status of a feature group export (pending, complete, etc.).
        """
        return self.describe().status

    # to be deleted
    def get_results(self):
        return self.client.get_export_result(self.feature_group_export_id)
