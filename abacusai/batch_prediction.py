from .batch_prediction_version import BatchPredictionVersion
from .prediction_input import PredictionInput
from .refresh_schedule import RefreshSchedule


class BatchPrediction():
    '''
        Batch predictions
    '''

    def __init__(self, client, batchPredictionId=None, createdAt=None, name=None, deploymentId=None, fileConnectorOutputLocation=None, globalPredictionArgs=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, featureGroupTableName=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, batchInputs={}, latestBatchPredictionVersion={}, refreshSchedules={}):
        self.client = client
        self.id = batchPredictionId
        self.batch_prediction_id = batchPredictionId
        self.created_at = createdAt
        self.name = name
        self.deployment_id = deploymentId
        self.file_connector_output_location = fileConnectorOutputLocation
        self.global_prediction_args = globalPredictionArgs
        self.database_connector_id = databaseConnectorId
        self.database_output_configuration = databaseOutputConfiguration
        self.explanations = explanations
        self.file_output_format = fileOutputFormat
        self.connector_type = connectorType
        self.legacy_input_location = legacyInputLocation
        self.feature_group_table_name = featureGroupTableName
        self.csv_input_prefix = csvInputPrefix
        self.csv_prediction_prefix = csvPredictionPrefix
        self.csv_explanations_prefix = csvExplanationsPrefix
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)
        self.latest_batch_prediction_version = client._build_class(
            BatchPredictionVersion, latestBatchPredictionVersion)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)

    def __repr__(self):
        return f"BatchPrediction(batch_prediction_id={repr(self.batch_prediction_id)}, created_at={repr(self.created_at)}, name={repr(self.name)}, deployment_id={repr(self.deployment_id)}, file_connector_output_location={repr(self.file_connector_output_location)}, global_prediction_args={repr(self.global_prediction_args)}, database_connector_id={repr(self.database_connector_id)}, database_output_configuration={repr(self.database_output_configuration)}, explanations={repr(self.explanations)}, file_output_format={repr(self.file_output_format)}, connector_type={repr(self.connector_type)}, legacy_input_location={repr(self.legacy_input_location)}, feature_group_table_name={repr(self.feature_group_table_name)}, csv_input_prefix={repr(self.csv_input_prefix)}, csv_prediction_prefix={repr(self.csv_prediction_prefix)}, csv_explanations_prefix={repr(self.csv_explanations_prefix)}, batch_inputs={repr(self.batch_inputs)}, latest_batch_prediction_version={repr(self.latest_batch_prediction_version)}, refresh_schedules={repr(self.refresh_schedules)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'batch_prediction_id': self.batch_prediction_id, 'created_at': self.created_at, 'name': self.name, 'deployment_id': self.deployment_id, 'file_connector_output_location': self.file_connector_output_location, 'global_prediction_args': self.global_prediction_args, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'feature_group_table_name': self.feature_group_table_name, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix': self.csv_explanations_prefix, 'batch_inputs': self.batch_inputs.to_dict() if self.batch_inputs else None, 'latest_batch_prediction_version': self.latest_batch_prediction_version.to_dict() if self.latest_batch_prediction_version else None, 'refresh_schedules': self.refresh_schedules.to_dict() if self.refresh_schedules else None}

    def start(self):
        return self.client.start_batch_prediction(self.batch_prediction_id)

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_batch_prediction(self.batch_prediction_id)

    def list_versions(self, limit=100, start_after_version=None):
        return self.client.list_batch_prediction_versions(self.batch_prediction_id, limit, start_after_version)

    def update(self, deployment_id=None, global_prediction_args=None, explanations=None, output_format=None, csv_input_prefix=None, csv_prediction_prefix=None, csv_explanations_prefix=None):
        return self.client.update_batch_prediction(self.batch_prediction_id, deployment_id, global_prediction_args, explanations, output_format, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix)

    def set_file_connector_output(self, output_format=None, output_location=None):
        return self.client.set_batch_prediction_file_connector_output(self.batch_prediction_id, output_format, output_location)

    def set_database_connector_output(self, database_connector_id=None, database_output_config=None):
        return self.client.set_batch_prediction_database_connector_output(self.batch_prediction_id, database_connector_id, database_output_config)

    def set_feature_group_output(self, table_name):
        return self.client.set_batch_prediction_feature_group_output(self.batch_prediction_id, table_name)

    def set_output_to_console(self):
        return self.client.set_batch_prediction_output_to_console(self.batch_prediction_id)

    def set_dataset(self, dataset_type, dataset_id=None):
        return self.client.set_batch_prediction_dataset(self.batch_prediction_id, dataset_type, dataset_id)

    def set_feature_group(self, feature_group_type, feature_group_id=None):
        return self.client.set_batch_prediction_feature_group(self.batch_prediction_id, feature_group_type, feature_group_id)

    def set_dataset_remap(self, dataset_id_remap):
        return self.client.set_batch_prediction_dataset_remap(self.batch_prediction_id, dataset_id_remap)

    def delete(self):
        return self.client.delete_batch_prediction(self.batch_prediction_id)

    def wait_for_predictions(self, timeout=1200):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().latest_batch_prediction_version.status
