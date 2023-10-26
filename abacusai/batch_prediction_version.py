from .api_class import BatchPredictionArgs
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
            driftMonitorStatus (str): The status of the drift monitor for this batch prediction version
            deploymentId (str): The deployment used to make the predictions
            modelId (str): The model used to make the predictions
            modelVersion (str): The model version used to make the predictions
            predictionsStartedAt (str): Predictions start date and time
            predictionsCompletedAt (str): Predictions completion date and time
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
            driftMonitorError (str): Error message for the drift monitor of this batch predcition
            monitorWarnings (str): Relevant warning if there are issues found in drift or data integrity
            csvInputPrefix (str): A prefix to prepend to the input columns, only applies when output format is CSV
            csvPredictionPrefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV
            csvExplanationsPrefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV
            databaseOutputTotalWrites (int): The total number of rows attempted to write (may be less than total_predictions if write mode is UPSERT and multiple rows share the same ID)
            databaseOutputFailedWrites (int): The number of failed writes to the Database Connector
            outputIncludesMetadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version
            resultInputColumns (list[str]): If present, will limit result files or feature groups to only include columns present in this list
            modelMonitorVersion (str): The version of the model monitor
            algoName (str): The name of the algorithm used to train the model
            algorithm (str): The algorithm that is currently deployed.
            outputFeatureGroupId (str): The BP output feature group id if applicable
            outputFeatureGroupVersion (str): The BP output feature group version if applicable
            outputFeatureGroupTableName (str): The BP output feature group name if applicable
            batchPredictionWarnings (str): Relevant warnings if any issues are found
            batchInputs (PredictionInput): Inputs to the batch prediction
    """

    def __init__(self, client, batchPredictionVersion=None, batchPredictionId=None, status=None, driftMonitorStatus=None, deploymentId=None, modelId=None, modelVersion=None, predictionsStartedAt=None, predictionsCompletedAt=None, databaseOutputError=None, totalPredictions=None, failedPredictions=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileConnectorOutputLocation=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, error=None, driftMonitorError=None, monitorWarnings=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, databaseOutputTotalWrites=None, databaseOutputFailedWrites=None, outputIncludesMetadata=None, resultInputColumns=None, modelMonitorVersion=None, algoName=None, algorithm=None, outputFeatureGroupId=None, outputFeatureGroupVersion=None, outputFeatureGroupTableName=None, batchPredictionWarnings=None, batchInputs={}, globalPredictionArgs={}):
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
        self.output_feature_group_id = outputFeatureGroupId
        self.output_feature_group_version = outputFeatureGroupVersion
        self.output_feature_group_table_name = outputFeatureGroupTableName
        self.batch_prediction_warnings = batchPredictionWarnings
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)
        self.global_prediction_args = client._build_class(
            BatchPredictionArgs, globalPredictionArgs)

    def __repr__(self):
        repr_dict = {f'batch_prediction_version': repr(self.batch_prediction_version), f'batch_prediction_id': repr(self.batch_prediction_id), f'status': repr(self.status), f'drift_monitor_status': repr(self.drift_monitor_status), f'deployment_id': repr(self.deployment_id), f'model_id': repr(self.model_id), f'model_version': repr(self.model_version), f'predictions_started_at': repr(self.predictions_started_at), f'predictions_completed_at': repr(self.predictions_completed_at), f'database_output_error': repr(self.database_output_error), f'total_predictions': repr(self.total_predictions), f'failed_predictions': repr(self.failed_predictions), f'database_connector_id': repr(self.database_connector_id), f'database_output_configuration': repr(self.database_output_configuration), f'explanations': repr(self.explanations), f'file_connector_output_location': repr(self.file_connector_output_location), f'file_output_format': repr(self.file_output_format), f'connector_type': repr(self.connector_type), f'legacy_input_location': repr(self.legacy_input_location), f'error': repr(
            self.error), f'drift_monitor_error': repr(self.drift_monitor_error), f'monitor_warnings': repr(self.monitor_warnings), f'csv_input_prefix': repr(self.csv_input_prefix), f'csv_prediction_prefix': repr(self.csv_prediction_prefix), f'csv_explanations_prefix': repr(self.csv_explanations_prefix), f'database_output_total_writes': repr(self.database_output_total_writes), f'database_output_failed_writes': repr(self.database_output_failed_writes), f'output_includes_metadata': repr(self.output_includes_metadata), f'result_input_columns': repr(self.result_input_columns), f'model_monitor_version': repr(self.model_monitor_version), f'algo_name': repr(self.algo_name), f'algorithm': repr(self.algorithm), f'output_feature_group_id': repr(self.output_feature_group_id), f'output_feature_group_version': repr(self.output_feature_group_version), f'output_feature_group_table_name': repr(self.output_feature_group_table_name), f'batch_prediction_warnings': repr(self.batch_prediction_warnings), f'batch_inputs': repr(self.batch_inputs), f'global_prediction_args': repr(self.global_prediction_args)}
        class_name = "BatchPredictionVersion"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'batch_prediction_version': self.batch_prediction_version, 'batch_prediction_id': self.batch_prediction_id, 'status': self.status, 'drift_monitor_status': self.drift_monitor_status, 'deployment_id': self.deployment_id, 'model_id': self.model_id, 'model_version': self.model_version, 'predictions_started_at': self.predictions_started_at, 'predictions_completed_at': self.predictions_completed_at, 'database_output_error': self.database_output_error, 'total_predictions': self.total_predictions, 'failed_predictions': self.failed_predictions, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_connector_output_location': self.file_connector_output_location, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'error': self.error, 'drift_monitor_error': self.drift_monitor_error,
                'monitor_warnings': self.monitor_warnings, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix': self.csv_explanations_prefix, 'database_output_total_writes': self.database_output_total_writes, 'database_output_failed_writes': self.database_output_failed_writes, 'output_includes_metadata': self.output_includes_metadata, 'result_input_columns': self.result_input_columns, 'model_monitor_version': self.model_monitor_version, 'algo_name': self.algo_name, 'algorithm': self.algorithm, 'output_feature_group_id': self.output_feature_group_id, 'output_feature_group_version': self.output_feature_group_version, 'output_feature_group_table_name': self.output_feature_group_table_name, 'batch_prediction_warnings': self.batch_prediction_warnings, 'batch_inputs': self._get_attribute_as_dict(self.batch_inputs), 'global_prediction_args': self._get_attribute_as_dict(self.global_prediction_args)}
        return {key: value for key, value in resp.items() if value is not None}

    def download_batch_prediction_result_chunk(self, offset: int = 0, chunk_size: int = 10485760):
        """
        Returns a stream containing the batch prediction results.

        Args:
            offset (int): The offset to read from.
            chunk_size (int): The maximum amount of data to read.
        """
        return self.client.download_batch_prediction_result_chunk(self.batch_prediction_version, offset, chunk_size)

    def get_batch_prediction_connector_errors(self):
        """
        Returns a stream containing the batch prediction database connection write errors, if any writes failed for the specified batch prediction job.

        Args:
            batch_prediction_version (str): Unique string identifier of the batch prediction job to get the errors for.
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
        Describes a Batch Prediction Version.

        Args:
            batch_prediction_version (str): Unique string identifier of the Batch Prediction Version.

        Returns:
            BatchPredictionVersion: The Batch Prediction Version.
        """
        return self.client.describe_batch_prediction_version(self.batch_prediction_version)

    def get_logs(self):
        """
        Retrieves the batch prediction logs.

        Args:
            batch_prediction_version (str): The unique version ID of the batch prediction version.

        Returns:
            BatchPredictionVersionLogs: The logs for the specified batch prediction version.
        """
        return self.client.get_batch_prediction_version_logs(self.batch_prediction_version)

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

    def wait_for_predictions(self, timeout=86400):
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
