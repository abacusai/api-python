from .prediction_input import PredictionInput


class BatchPredictionVersion():
    '''

    '''

    def __init__(self, client, batchPredictionVersion=None, batchPredictionId=None, status=None, deploymentId=None, modelVersion=None, predictionsStartedAt=None, predictionsCompletedAt=None, globalPredictionArgs=None, totalPredictions=None, failedPredictions=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileConnectorOutputLocation=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, batchInputs={}):
        self.client = client
        self.id = batchPredictionVersion
        self.batch_prediction_version = batchPredictionVersion
        self.batch_prediction_id = batchPredictionId
        self.status = status
        self.deployment_id = deploymentId
        self.model_version = modelVersion
        self.predictions_started_at = predictionsStartedAt
        self.predictions_completed_at = predictionsCompletedAt
        self.global_prediction_args = globalPredictionArgs
        self.total_predictions = totalPredictions
        self.failed_predictions = failedPredictions
        self.database_connector_id = databaseConnectorId
        self.database_output_configuration = databaseOutputConfiguration
        self.explanations = explanations
        self.file_connector_output_location = fileConnectorOutputLocation
        self.file_output_format = fileOutputFormat
        self.connector_type = connectorType
        self.legacy_input_location = legacyInputLocation
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)

    def __repr__(self):
        return f"BatchPredictionVersion(batch_prediction_version={repr(self.batch_prediction_version)}, batch_prediction_id={repr(self.batch_prediction_id)}, status={repr(self.status)}, deployment_id={repr(self.deployment_id)}, model_version={repr(self.model_version)}, predictions_started_at={repr(self.predictions_started_at)}, predictions_completed_at={repr(self.predictions_completed_at)}, global_prediction_args={repr(self.global_prediction_args)}, total_predictions={repr(self.total_predictions)}, failed_predictions={repr(self.failed_predictions)}, database_connector_id={repr(self.database_connector_id)}, database_output_configuration={repr(self.database_output_configuration)}, explanations={repr(self.explanations)}, file_connector_output_location={repr(self.file_connector_output_location)}, file_output_format={repr(self.file_output_format)}, connector_type={repr(self.connector_type)}, legacy_input_location={repr(self.legacy_input_location)}, batch_inputs={repr(self.batch_inputs)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'batch_prediction_version': self.batch_prediction_version, 'batch_prediction_id': self.batch_prediction_id, 'status': self.status, 'deployment_id': self.deployment_id, 'model_version': self.model_version, 'predictions_started_at': self.predictions_started_at, 'predictions_completed_at': self.predictions_completed_at, 'global_prediction_args': self.global_prediction_args, 'total_predictions': self.total_predictions, 'failed_predictions': self.failed_predictions, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_connector_output_location': self.file_connector_output_location, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'batch_inputs': [elem.to_dict() for elem in self.batch_inputs or []]}
