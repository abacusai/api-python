from .prediction_input import PredictionInput
from .return_class import AbstractApiClass


class BatchPredictionVersion(AbstractApiClass):
    """
        Batch Prediction Version
    """

    def __init__(self, client, batchPredictionVersion=None, batchPredictionId=None, status=None, deploymentId=None, modelId=None, modelVersion=None, predictionsStartedAt=None, predictionsCompletedAt=None, globalPredictionArgs=None, totalPredictions=None, failedPredictions=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileConnectorOutputLocation=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, error=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, batchInputs={}):
        super().__init__(client, batchPredictionVersion)
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
        return f"BatchPredictionVersion(batch_prediction_version={repr(self.batch_prediction_version)},\n  batch_prediction_id={repr(self.batch_prediction_id)},\n  status={repr(self.status)},\n  deployment_id={repr(self.deployment_id)},\n  model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  predictions_started_at={repr(self.predictions_started_at)},\n  predictions_completed_at={repr(self.predictions_completed_at)},\n  global_prediction_args={repr(self.global_prediction_args)},\n  total_predictions={repr(self.total_predictions)},\n  failed_predictions={repr(self.failed_predictions)},\n  database_connector_id={repr(self.database_connector_id)},\n  database_output_configuration={repr(self.database_output_configuration)},\n  explanations={repr(self.explanations)},\n  file_connector_output_location={repr(self.file_connector_output_location)},\n  file_output_format={repr(self.file_output_format)},\n  connector_type={repr(self.connector_type)},\n  legacy_input_location={repr(self.legacy_input_location)},\n  error={repr(self.error)},\n  csv_input_prefix={repr(self.csv_input_prefix)},\n  csv_prediction_prefix={repr(self.csv_prediction_prefix)},\n  csv_explanations_prefix={repr(self.csv_explanations_prefix)},\n  batch_inputs={repr(self.batch_inputs)})"

    def to_dict(self):
        return {'batch_prediction_version': self.batch_prediction_version, 'batch_prediction_id': self.batch_prediction_id, 'status': self.status, 'deployment_id': self.deployment_id, 'model_id': self.model_id, 'model_version': self.model_version, 'predictions_started_at': self.predictions_started_at, 'predictions_completed_at': self.predictions_completed_at, 'global_prediction_args': self.global_prediction_args, 'total_predictions': self.total_predictions, 'failed_predictions': self.failed_predictions, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_connector_output_location': self.file_connector_output_location, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'error': self.error, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix': self.csv_explanations_prefix, 'batch_inputs': self._get_attribute_as_dict(self.batch_inputs)}

    def download_batch_prediction_result_chunk(self, offset=0, chunk_size=10485760):
        """Returns a stream containing the batch prediction results"""
        return self.client.download_batch_prediction_result_chunk(self.batch_prediction_version, offset, chunk_size)

    def get_batch_prediction_connector_errors(self):
        """Returns a stream containing the batch prediction database connection write errors, if any writes failed to the database connector"""
        return self.client.get_batch_prediction_connector_errors(self.batch_prediction_version)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Describes a batch prediction version"""
        return self.client.describe_batch_prediction_version(self.batch_prediction_version)

    def download_result_to_file(self, file):
        """
        Downloads the batch prediction version in a local file.

        Args:
            file (file object): A file object opened in a binary mode e.g., file=open('/tmp/output', 'wb').

        Returns:
            None
        """
        offset = 0
        while True:
            with self.download_batch_prediction_result_chunk(offset) as chunk:
                bytes_written = file.write(chunk.read())
            if not bytes_written:
                break
            offset += bytes_written

    def wait_for_predictions(self, timeout=1200):
        """
        A waiting call until batch prediction version is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 1200 milliseconds.

        Returns:
            None
        """
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the batch prediction version.

        Returns:
            Enum (string): A string describing the status of the batch prediction version, for e.g., pending, complete, etc.
            """
        return self.describe().status
