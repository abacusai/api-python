from .batch_prediction_version import BatchPredictionVersion
from .prediction_input import PredictionInput


class BatchPrediction():
    '''

    '''

    def __init__(self, client, batchPredictionId=None, createdAt=None, name=None, deploymentId=None, fileConnectorOutputLocation=None, globalPredictionArgs=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, refreshSchedules=None, batchInputs={}, latestBatchPredictionVersion={}):
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
        self.refresh_schedules = refreshSchedules
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)
        self.latest_batch_prediction_version = client._build_class(
            BatchPredictionVersion, latestBatchPredictionVersion)

    def __repr__(self):
        return f"BatchPrediction(batch_prediction_id={repr(self.batch_prediction_id)}, created_at={repr(self.created_at)}, name={repr(self.name)}, deployment_id={repr(self.deployment_id)}, file_connector_output_location={repr(self.file_connector_output_location)}, global_prediction_args={repr(self.global_prediction_args)}, database_connector_id={repr(self.database_connector_id)}, database_output_configuration={repr(self.database_output_configuration)}, explanations={repr(self.explanations)}, file_output_format={repr(self.file_output_format)}, connector_type={repr(self.connector_type)}, legacy_input_location={repr(self.legacy_input_location)}, refresh_schedules={repr(self.refresh_schedules)}, batch_inputs={repr(self.batch_inputs)}, latest_batch_prediction_version={repr(self.latest_batch_prediction_version)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'batch_prediction_id': self.batch_prediction_id, 'created_at': self.created_at, 'name': self.name, 'deployment_id': self.deployment_id, 'file_connector_output_location': self.file_connector_output_location, 'global_prediction_args': self.global_prediction_args, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'refresh_schedules': self.refresh_schedules, 'batch_inputs': [elem.to_dict() for elem in self.batch_inputs or []], 'latest_batch_prediction_version': [elem.to_dict() for elem in self.latest_batch_prediction_version or []]}

    def start(self):
        return self.client.start_batch_prediction(self.batch_prediction_id)

    def refresh(self):
        self = self.describe()
        return self

    def describe(self):
        return self.client.describe_batch_prediction(self.batch_prediction_id)

    def list_versions(self):
        return self.client.list_batch_prediction_versions(self.batch_prediction_id)

    def update(self, deployment_id=None, global_prediction_args=None, explanations=None, output_format=None):
        return self.client.update_batch_prediction(self.batch_prediction_id, deployment_id, global_prediction_args, explanations, output_format)

    def set_file_connector_output(self, output_format=None, output_location=None):
        return self.client.set_batch_prediction_file_connector_output(self.batch_prediction_id, output_format, output_location)

    def set_database_connector_output(self, database_connector_id=None, database_output_config=None):
        return self.client.set_batch_prediction_database_connector_output(self.batch_prediction_id, database_connector_id, database_output_config)

    def set_output_to_console(self):
        return self.client.set_batch_prediction_output_to_console(self.batch_prediction_id)

    def set_dataset(self, dataset_type, dataset_id=None):
        return self.client.set_batch_prediction_dataset(self.batch_prediction_id, dataset_type, dataset_id)

    def set_feature_group(self, dataset_type, feature_group_id=None):
        return self.client.set_batch_prediction_feature_group(self.batch_prediction_id, dataset_type, feature_group_id)

    def delete(self):
        return self.client.delete_batch_prediction(self.batch_prediction_id)

    def wait_for_predictions(self, timeout=1200):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().latest_batch_prediction_version.status
