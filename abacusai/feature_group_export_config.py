from .api_class import FeatureGroupExportConfig
from .return_class import AbstractApiClass


class FeatureGroupExportConfig(AbstractApiClass):
    """
        Export configuration (file connector or database connector information) for feature group exports.

        Args:
            client (ApiClient): An authenticated API Client instance
            outputLocation (str): The File Connector location to which the feature group is being written.
            fileFormat (str): The file format being written to output_location.
            databaseConnectorId (str): The unique string identifier of the database connector used.
            objectName (str): The object in the database connector to which the feature group is being written.
            writeMode (str): UPSERT or INSERT for writing to the database connector.
            databaseFeatureMapping (dict): The column/feature pairs mapping the features to the database columns.
            idColumn (str): The id column to use as the upsert key.
            additionalIdColumns (str): For database connectors which support it, additional ID columns to use as a complex key for upserting.
    """

    def __init__(self, client, outputLocation=None, fileFormat=None, databaseConnectorId=None, objectName=None, writeMode=None, databaseFeatureMapping=None, idColumn=None, additionalIdColumns=None):
        super().__init__(client, None)
        self.output_location = outputLocation
        self.file_format = fileFormat
        self.database_connector_id = databaseConnectorId
        self.object_name = objectName
        self.write_mode = writeMode
        self.database_feature_mapping = databaseFeatureMapping
        self.id_column = idColumn
        self.additional_id_columns = additionalIdColumns

    def __repr__(self):
        return f"FeatureGroupExportConfig(output_location={repr(self.output_location)},\n  file_format={repr(self.file_format)},\n  database_connector_id={repr(self.database_connector_id)},\n  object_name={repr(self.object_name)},\n  write_mode={repr(self.write_mode)},\n  database_feature_mapping={repr(self.database_feature_mapping)},\n  id_column={repr(self.id_column)},\n  additional_id_columns={repr(self.additional_id_columns)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'output_location': self.output_location, 'file_format': self.file_format, 'database_connector_id': self.database_connector_id, 'object_name': self.object_name, 'write_mode': self.write_mode, 'database_feature_mapping': self.database_feature_mapping, 'id_column': self.id_column, 'additional_id_columns': self.additional_id_columns}
