from .prediction_input import PredictionInput
from .return_class import AbstractApiClass


class BatchPredictionVersion(AbstractApiClass):
    """
        Batch Prediction Version

        Args:
            client (ApiClient): An authenticated API Client instance
            batchPredictionVersion (str): The unique identifier of the batch prediction
            batchPredictionId (str): The unique identifier of the batch prediction
            status (str): The current status of the batch prediction
            driftMonitorStatus (str): 
            deploymentId (str): The deployment used to make the predictions
            modelId (str): The model used to make the predictions
            modelVersion (str): The model version used to make the predictions
            predictionsStartedAt (str): Predictions start date and time
            predictionsCompletedAt (str): Predictions completion date and time
            globalPredictionArgs (dict): Argument(s) passed to every prediction call
            databaseOutputError (bool): If true, there were errors reported by the database connector while writing
            totalPredictions (int): Number of predictions performed in this batch prediction job
            failedPredictions (int): Number of predictions that failed
            databaseConnectorId (str): The database connector to write the results to
            databaseOutputConfiguration (dict): Contains information about where the batch predictions are written to
            explanations (bool): If true, explanations for each prediction were created
            fileConnectorOutputLocation (str):  Contains information about where the batch predictions are written to
            fileOutputFormat (str): The format of the batch prediction output (CSV or JSON)
            connectorType (str): Null if writing to internal console, else FEATURE_GROUP | FILE_CONNECTOR | DATABASE_CONNECTOR
            legacyInputLocation (str): The location of the input data
            error (str): Relevant error if the status is FAILED
            driftMonitorError (str): 
            monitorWarnings (str): Relevant warning if there are issues found in drift or data integrity
            csvInputPrefix (str): A prefix to prepend to the input columns, only applies when output format is CSV
            csvPredictionPrefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV
            csvExplanationsPrefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV
            databaseOutputTotalWrites (int): The total number of rows attempted to write (may be less than total_predictions if write mode is UPSERT and multiple rows share the same ID)
            databaseOutputFailedWrites (int): The number of failed writes to the Database Connector
            outputIncludesMetadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version
            resultInputColumns (list of string): If present, will limit result files or feature groups to only include columns present in this list
            modelMonitorVersion (str): 
            algoName (str): The name of the algorithm used to train the model
            algorithm (str): The algorithm that is currently deployed.
            batchInputs (PredictionInput): Inputs to the batch prediction
    """

    def __init__(self, client, batchPredictionVersion=None, batchPredictionId=None, status=None, driftMonitorStatus=None, deploymentId=None, modelId=None, modelVersion=None, predictionsStartedAt=None, predictionsCompletedAt=None, globalPredictionArgs=None, databaseOutputError=None, totalPredictions=None, failedPredictions=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileConnectorOutputLocation=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, error=None, driftMonitorError=None, monitorWarnings=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, databaseOutputTotalWrites=None, databaseOutputFailedWrites=None, outputIncludesMetadata=None, resultInputColumns=None, modelMonitorVersion=None, algoName=None, algorithm=None, batchInputs={}):
        super().__init__(client, batchPredictionVersion)
        self.batch_prediction_version = batchPredictionVersion
        self.batch_prediction_id = batchPredictionId
        self.status = status
        self.drift_monitor_status = driftMonitorStatus
        self.deployment_id = deploymentId
        self.model_id = modelId
        self.model_version = modelVersion
        self.predictions_started_at = predictionsStartedAt
        self.predictions_completed_at = predictionsCompletedAt
        self.global_prediction_args = globalPredictionArgs
        self.database_output_error = databaseOutputError
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
        self.drift_monitor_error = driftMonitorError
        self.monitor_warnings = monitorWarnings
        self.csv_input_prefix = csvInputPrefix
        self.csv_prediction_prefix = csvPredictionPrefix
        self.csv_explanations_prefix = csvExplanationsPrefix
        self.database_output_total_writes = databaseOutputTotalWrites
        self.database_output_failed_writes = databaseOutputFailedWrites
        self.output_includes_metadata = outputIncludesMetadata
        self.result_input_columns = resultInputColumns
        self.model_monitor_version = modelMonitorVersion
        self.algo_name = algoName
        self.algorithm = algorithm
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)

    def __repr__(self):
        return f"BatchPredictionVersion(batch_prediction_version={repr(self.batch_prediction_version)},\n  batch_prediction_id={repr(self.batch_prediction_id)},\n  status={repr(self.status)},\n  drift_monitor_status={repr(self.drift_monitor_status)},\n  deployment_id={repr(self.deployment_id)},\n  model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  predictions_started_at={repr(self.predictions_started_at)},\n  predictions_completed_at={repr(self.predictions_completed_at)},\n  global_prediction_args={repr(self.global_prediction_args)},\n  database_output_error={repr(self.database_output_error)},\n  total_predictions={repr(self.total_predictions)},\n  failed_predictions={repr(self.failed_predictions)},\n  database_connector_id={repr(self.database_connector_id)},\n  database_output_configuration={repr(self.database_output_configuration)},\n  explanations={repr(self.explanations)},\n  file_connector_output_location={repr(self.file_connector_output_location)},\n  file_output_format={repr(self.file_output_format)},\n  connector_type={repr(self.connector_type)},\n  legacy_input_location={repr(self.legacy_input_location)},\n  error={repr(self.error)},\n  drift_monitor_error={repr(self.drift_monitor_error)},\n  monitor_warnings={repr(self.monitor_warnings)},\n  csv_input_prefix={repr(self.csv_input_prefix)},\n  csv_prediction_prefix={repr(self.csv_prediction_prefix)},\n  csv_explanations_prefix={repr(self.csv_explanations_prefix)},\n  database_output_total_writes={repr(self.database_output_total_writes)},\n  database_output_failed_writes={repr(self.database_output_failed_writes)},\n  output_includes_metadata={repr(self.output_includes_metadata)},\n  result_input_columns={repr(self.result_input_columns)},\n  model_monitor_version={repr(self.model_monitor_version)},\n  algo_name={repr(self.algo_name)},\n  algorithm={repr(self.algorithm)},\n  batch_inputs={repr(self.batch_inputs)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'batch_prediction_version': self.batch_prediction_version, 'batch_prediction_id': self.batch_prediction_id, 'status': self.status, 'drift_monitor_status': self.drift_monitor_status, 'deployment_id': self.deployment_id, 'model_id': self.model_id, 'model_version': self.model_version, 'predictions_started_at': self.predictions_started_at, 'predictions_completed_at': self.predictions_completed_at, 'global_prediction_args': self.global_prediction_args, 'database_output_error': self.database_output_error, 'total_predictions': self.total_predictions, 'failed_predictions': self.failed_predictions, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_connector_output_location': self.file_connector_output_location, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'error': self.error, 'drift_monitor_error': self.drift_monitor_error, 'monitor_warnings': self.monitor_warnings, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix': self.csv_explanations_prefix, 'database_output_total_writes': self.database_output_total_writes, 'database_output_failed_writes': self.database_output_failed_writes, 'output_includes_metadata': self.output_includes_metadata, 'result_input_columns': self.result_input_columns, 'model_monitor_version': self.model_monitor_version, 'algo_name': self.algo_name, 'algorithm': self.algorithm, 'batch_inputs': self._get_attribute_as_dict(self.batch_inputs)}

    def get_outliers_for_batch_prediction_feature(self, feature_name: str = None):
        """
        Gets a list of outliers measured by a single feature (or overall) in an output feature group from a prediction.

        Args:
            feature_name (str): Name of the feature to view the distribution of.
        """
        return self.client.get_outliers_for_batch_prediction_feature(self.batch_prediction_version, feature_name)

    def download_batch_prediction_result_chunk(self, offset: int = 0, chunk_size: int = 10485760):
        """
        Returns a stream containing the batch prediction results

        Args:
            offset (int): The offset to read from
            chunk_size (int): The max amount of data to read
        """
        return self.client.download_batch_prediction_result_chunk(self.batch_prediction_version, offset, chunk_size)

    def get_batch_prediction_connector_errors(self):
        """
        Returns a stream containing the batch prediction database connection write errors, if any writes failed to the database connector

        Args:
            batch_prediction_version (str): The unique identifier of the batch prediction job to get the errors for
        """
        return self.client.get_batch_prediction_connector_errors(self.batch_prediction_version)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            BatchPredictionVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes a batch prediction version

        Args:
            batch_prediction_version (str): The unique identifier of the batch prediction version

        Returns:
            BatchPredictionVersion: The batch prediction version.
        """
        return self.client.describe_batch_prediction_version(self.batch_prediction_version)

    def download_result_to_file(self, file):
        """
        Downloads the batch prediction version in a local file.

        Args:
            file (file object): A file object opened in a binary mode e.g., file=open('/tmp/output', 'wb').
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
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def wait_for_drift_monitor(self, timeout=86400):
        """
        A waiting call until batch prediction drift monitor calculations are ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'MONITORING'}, poll_args={'drift_monitor_status': True}, timeout=timeout)

    def get_status(self, drift_monitor_status: bool = False):
        """
        Gets the status of the batch prediction version.

        Returns:
            str: A string describing the status of the batch prediction version, for e.g., pending, complete, etc.
            """
        if drift_monitor_status:
            return self.describe().drift_monitor_status
        return self.describe().status
