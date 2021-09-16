

class FeatureGroupExport():
    '''
        A feature Group Export Job
    '''

    def __init__(self, client, featureGroupExportId=None, featureGroupVersion=None, connectorType=None, outputLocation=None, fileFormat=None, objectName=None, writeMode=None, databaseFeatureMapping=None, idColumn=None, status=None, createdAt=None, exportCompletedAt=None):
        self.client = client
        self.id = featureGroupExportId
        self.feature_group_export_id = featureGroupExportId
        self.feature_group_version = featureGroupVersion
        self.connector_type = connectorType
        self.output_location = outputLocation
        self.file_format = fileFormat
        self.object_name = objectName
        self.write_mode = writeMode
        self.database_feature_mapping = databaseFeatureMapping
        self.id_column = idColumn
        self.status = status
        self.created_at = createdAt
        self.export_completed_at = exportCompletedAt

    def __repr__(self):
        return f"FeatureGroupExport(feature_group_export_id={repr(self.feature_group_export_id)}, feature_group_version={repr(self.feature_group_version)}, connector_type={repr(self.connector_type)}, output_location={repr(self.output_location)}, file_format={repr(self.file_format)}, object_name={repr(self.object_name)}, write_mode={repr(self.write_mode)}, database_feature_mapping={repr(self.database_feature_mapping)}, id_column={repr(self.id_column)}, status={repr(self.status)}, created_at={repr(self.created_at)}, export_completed_at={repr(self.export_completed_at)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'feature_group_export_id': self.feature_group_export_id, 'feature_group_version': self.feature_group_version, 'connector_type': self.connector_type, 'output_location': self.output_location, 'file_format': self.file_format, 'object_name': self.object_name, 'write_mode': self.write_mode, 'database_feature_mapping': self.database_feature_mapping, 'id_column': self.id_column, 'status': self.status, 'created_at': self.created_at, 'export_completed_at': self.export_completed_at}

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_feature_group_export(self.feature_group_export_id)

    def describe(self):
        return self.client.describe_feature_group_export(self.feature_group_export_id)

    def wait_for_results(self, timeout=3600):
        return self.client._poll(self, {'PENDING', 'EXPORTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status

    def get_results(self):
        return self.client.get_export_result(self.feature_group_export_id)
