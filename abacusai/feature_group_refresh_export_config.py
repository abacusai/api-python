from .return_class import AbstractApiClass


class FeatureGroupRefreshExportConfig(AbstractApiClass):
    """
        A Feature Group Refresh Export Config outlines the export configuration for a feature group.

        Args:
            client (ApiClient): An authenticated API Client instance
            connectorType (str): The type of connector the feature group is
            location (str): The file connector location of the feature group export
            exportFileFormat (str): The file format of the feature group export
            additionalIdColumns (list): Additional id columns to use for upsert operations
            databaseFeatureMapping (dict): The mapping of feature names to database columns
            externalConnectionId (str): The unique identifier of the external connection to write to
            idColumn (str): The column to use as the id column for upsert operations
            objectName (str): The name of the object to write to
            writeMode (str): The write mode to use for the export
    """

    def __init__(self, client, connectorType=None, location=None, exportFileFormat=None, additionalIdColumns=None, databaseFeatureMapping=None, externalConnectionId=None, idColumn=None, objectName=None, writeMode=None):
        super().__init__(client, None)
        self.connector_type = connectorType
        self.location = location
        self.export_file_format = exportFileFormat
        self.additional_id_columns = additionalIdColumns
        self.database_feature_mapping = databaseFeatureMapping
        self.external_connection_id = externalConnectionId
        self.id_column = idColumn
        self.object_name = objectName
        self.write_mode = writeMode

    def __repr__(self):
        return f"FeatureGroupRefreshExportConfig(connector_type={repr(self.connector_type)},\n  location={repr(self.location)},\n  export_file_format={repr(self.export_file_format)},\n  additional_id_columns={repr(self.additional_id_columns)},\n  database_feature_mapping={repr(self.database_feature_mapping)},\n  external_connection_id={repr(self.external_connection_id)},\n  id_column={repr(self.id_column)},\n  object_name={repr(self.object_name)},\n  write_mode={repr(self.write_mode)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'connector_type': self.connector_type, 'location': self.location, 'export_file_format': self.export_file_format, 'additional_id_columns': self.additional_id_columns, 'database_feature_mapping': self.database_feature_mapping, 'external_connection_id': self.external_connection_id, 'id_column': self.id_column, 'object_name': self.object_name, 'write_mode': self.write_mode}
