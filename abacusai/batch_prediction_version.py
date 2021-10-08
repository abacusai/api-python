from .prediction_input import PredictionInput


class BatchPredictionVersion():
    '''
        Batch Prediction Version
    '''

    def __init__(self, client, batchPredictionVersion=None, batchPredictionId=None, status=None, deploymentId=None, modelId=None, modelVersion=None, predictionsStartedAt=None, predictionsCompletedAt=None, globalPredictionArgs=None, totalPredictions=None, failedPredictions=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileConnectorOutputLocation=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, error=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, batchInputs={}):
        self.client = client
        self.id = batchPredictionVersion
        self.batch_prediction_version = batchPredictionVersion
        self.batch_prediction_id = batchPredictionId
        self.status = status
        self.deployment_id = deploymentId
        self.model_id = modelId
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
        self.error = error
        self.csv_input_prefix = csvInputPrefix
        self.csv_prediction_prefix = csvPredictionPrefix
        self.csv_explanations_prefix = csvExplanationsPrefix
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)

    def __repr__(self):
        return f"BatchPredictionVersion(batch_prediction_version={repr(self.batch_prediction_version)}, batch_prediction_id={repr(self.batch_prediction_id)}, status={repr(self.status)}, deployment_id={repr(self.deployment_id)}, model_id={repr(self.model_id)}, model_version={repr(self.model_version)}, predictions_started_at={repr(self.predictions_started_at)}, predictions_completed_at={repr(self.predictions_completed_at)}, global_prediction_args={repr(self.global_prediction_args)}, total_predictions={repr(self.total_predictions)}, failed_predictions={repr(self.failed_predictions)}, database_connector_id={repr(self.database_connector_id)}, database_output_configuration={repr(self.database_output_configuration)}, explanations={repr(self.explanations)}, file_connector_output_location={repr(self.file_connector_output_location)}, file_output_format={repr(self.file_output_format)}, connector_type={repr(self.connector_type)}, legacy_input_location={repr(self.legacy_input_location)}, error={repr(self.error)}, csv_input_prefix={repr(self.csv_input_prefix)}, csv_prediction_prefix={repr(self.csv_prediction_prefix)}, csv_explanations_prefix={repr(self.csv_explanations_prefix)}, batch_inputs={repr(self.batch_inputs)})"

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def to_dict(self):
        return {'batch_prediction_version': self.batch_prediction_version, 'batch_prediction_id': self.batch_prediction_id, 'status': self.status, 'deployment_id': self.deployment_id, 'model_id': self.model_id, 'model_version': self.model_version, 'predictions_started_at': self.predictions_started_at, 'predictions_completed_at': self.predictions_completed_at, 'global_prediction_args': self.global_prediction_args, 'total_predictions': self.total_predictions, 'failed_predictions': self.failed_predictions, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_connector_output_location': self.file_connector_output_location, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'error': self.error, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix': self.csv_explanations_prefix, 'batch_inputs': self.batch_inputs.to_dict() if self.batch_inputs else None}

    def get_batch_prediction_result(self):
        return self.client.get_batch_prediction_result(self.batch_prediction_version)

    def download_batch_prediction_result_chunk(self, offset=0, chunk_size=10485760):
        return self.client.download_batch_prediction_result_chunk(self.batch_prediction_version, offset, chunk_size)

    def get_batch_prediction_connector_errors(self):
        return self.client.get_batch_prediction_connector_errors(self.batch_prediction_version)

    def refresh(self):
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        return self.client.describe_batch_prediction_version(self.batch_prediction_version)

    def download_result_to_file(self, file):
        offset = 0
        while True:
            with self.download_batch_prediction_result_chunk(offset) as chunk:
                bytes_written = file.write(chunk.read())
            if not bytes_written:
                break
            offset += bytes_written

    def wait_for_predictions(self, timeout=1200):
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def get_status(self):
        return self.describe().status
