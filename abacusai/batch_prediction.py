from .batch_prediction_version import BatchPredictionVersion
from .prediction_input import PredictionInput
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class BatchPrediction(AbstractApiClass):
    """
        Batch predictions

        Args:
            client (ApiClient): An authenticated API Client instance
            batchPredictionId (str): The unique identifier of the batch prediction request
            createdAt (str): When the batch prediction was created
            name (str): Name given to the batch prediction object
            deploymentId (str): The deployment used to make the predictions
            fileConnectorOutputLocation (str): Contains information about where the batch predictions are written to
            globalPredictionArgs (dict): Argument(s) passed to every prediction call
            databaseConnectorId (str): The database connector to write the results to
            databaseOutputConfiguration (dict): Contains information about where the batch predictions are written to
            explanations (bool): If true, explanations for each prediction were created
            fileOutputFormat (str): The format of the batch prediction output (CSV or JSON)
            connectorType (str): Null if writing to internal console, else FEATURE_GROUP | FILE_CONNECTOR | DATABASE_CONNECTOR
            legacyInputLocation (str): The location of the input data
            featureGroupTableName (str): The table name of the Batch Prediction feature group
            summaryFeatureGroupTableName (str): The table name of the metrics summary feature group output by Batch Prediction
            csvInputPrefix (str): A prefix to prepend to the input columns, only applies when output format is CSV
            csvPredictionPrefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV
            csvExplanationsPrefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV
            outputIncludesMetadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version
            resultInputColumns (list of string): If present, will limit result files or feature groups to only include columns present in this list
            modelMonitorId (str): 
            batchInputs (PredictionInput): Inputs to the batch prediction
            latestBatchPredictionVersion (BatchPredictionVersion): The latest batch prediction version
            refreshSchedules (RefreshSchedule): List of refresh schedules that dictate the next time the batch prediction will be run
    """

    def __init__(self, client, batchPredictionId=None, createdAt=None, name=None, deploymentId=None, fileConnectorOutputLocation=None, globalPredictionArgs=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, featureGroupTableName=None, summaryFeatureGroupTableName=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, outputIncludesMetadata=None, resultInputColumns=None, modelMonitorId=None, batchInputs={}, latestBatchPredictionVersion={}, refreshSchedules={}):
        super().__init__(client, batchPredictionId)
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
        self.summary_feature_group_table_name = summaryFeatureGroupTableName
        self.csv_input_prefix = csvInputPrefix
        self.csv_prediction_prefix = csvPredictionPrefix
        self.csv_explanations_prefix = csvExplanationsPrefix
        self.output_includes_metadata = outputIncludesMetadata
        self.result_input_columns = resultInputColumns
        self.model_monitor_id = modelMonitorId
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)
        self.latest_batch_prediction_version = client._build_class(
            BatchPredictionVersion, latestBatchPredictionVersion)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)

    def __repr__(self):
        return f"BatchPrediction(batch_prediction_id={repr(self.batch_prediction_id)},\n  created_at={repr(self.created_at)},\n  name={repr(self.name)},\n  deployment_id={repr(self.deployment_id)},\n  file_connector_output_location={repr(self.file_connector_output_location)},\n  global_prediction_args={repr(self.global_prediction_args)},\n  database_connector_id={repr(self.database_connector_id)},\n  database_output_configuration={repr(self.database_output_configuration)},\n  explanations={repr(self.explanations)},\n  file_output_format={repr(self.file_output_format)},\n  connector_type={repr(self.connector_type)},\n  legacy_input_location={repr(self.legacy_input_location)},\n  feature_group_table_name={repr(self.feature_group_table_name)},\n  summary_feature_group_table_name={repr(self.summary_feature_group_table_name)},\n  csv_input_prefix={repr(self.csv_input_prefix)},\n  csv_prediction_prefix={repr(self.csv_prediction_prefix)},\n  csv_explanations_prefix={repr(self.csv_explanations_prefix)},\n  output_includes_metadata={repr(self.output_includes_metadata)},\n  result_input_columns={repr(self.result_input_columns)},\n  model_monitor_id={repr(self.model_monitor_id)},\n  batch_inputs={repr(self.batch_inputs)},\n  latest_batch_prediction_version={repr(self.latest_batch_prediction_version)},\n  refresh_schedules={repr(self.refresh_schedules)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'batch_prediction_id': self.batch_prediction_id, 'created_at': self.created_at, 'name': self.name, 'deployment_id': self.deployment_id, 'file_connector_output_location': self.file_connector_output_location, 'global_prediction_args': self.global_prediction_args, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'feature_group_table_name': self.feature_group_table_name, 'summary_feature_group_table_name': self.summary_feature_group_table_name, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix': self.csv_explanations_prefix, 'output_includes_metadata': self.output_includes_metadata, 'result_input_columns': self.result_input_columns, 'model_monitor_id': self.model_monitor_id, 'batch_inputs': self._get_attribute_as_dict(self.batch_inputs), 'latest_batch_prediction_version': self._get_attribute_as_dict(self.latest_batch_prediction_version), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules)}

    def start(self):
        """
        Creates a new batch prediction version job for a given batch prediction job description

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction to create a new version of

        Returns:
            BatchPredictionVersion: The batch prediction version started by this method call.
        """
        return self.client.start_batch_prediction(self.batch_prediction_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            BatchPrediction: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Describes the batch prediction

        Args:
            batch_prediction_id (str): The unique ID associated with the batch prediction.

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.describe_batch_prediction(self.batch_prediction_id)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of versions of a given batch prediction

        Args:
            limit (int): The number of versions to list
            start_after_version (str): The version to start after

        Returns:
            BatchPredictionVersion: A list of batch prediction versions.
        """
        return self.client.list_batch_prediction_versions(self.batch_prediction_id, limit, start_after_version)

    def update(self, deployment_id: str = None, global_prediction_args: dict = None, explanations: bool = None, output_format: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None):
        """
        Updates a batch prediction job description

        Args:
            deployment_id (str): The unique identifier to a deployment.
            global_prediction_args (dict): Argument(s) to pass on every prediction call.
            explanations (bool): If true, will provide SHAP Explanations for each prediction, if supported by the use case.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON).
            csv_input_prefix (str): A prefix to prepend to the input columns, only applies when output format is CSV
            csv_prediction_prefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV
            csv_explanations_prefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV
            output_includes_metadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.update_batch_prediction(self.batch_prediction_id, deployment_id, global_prediction_args, explanations, output_format, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix, output_includes_metadata, result_input_columns)

    def set_file_connector_output(self, output_format: str = None, output_location: str = None):
        """
        Updates the file connector output configuration of the batch prediction

        Args:
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON).
            output_location (str): If specified, the location to write the prediction results. Otherwise, results will be stored in Abacus.AI.

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.set_batch_prediction_file_connector_output(self.batch_prediction_id, output_format, output_location)

    def set_database_connector_output(self, database_connector_id: str = None, database_output_config: dict = None):
        """
        Updates the database connector output configuration of the batch prediction

        Args:
            database_connector_id (str): The unique identifier of an Database Connection to write predictions to.
            database_output_config (dict): A key-value pair of columns/values to write to the database connector

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.set_batch_prediction_database_connector_output(self.batch_prediction_id, database_connector_id, database_output_config)

    def set_feature_group_output(self, table_name: str):
        """
        Creates a feature group and sets it to be the batch prediction output

        Args:
            table_name (str): The name of the feature group table to create

        Returns:
            BatchPrediction: The batch prediction after the output has been applied
        """
        return self.client.set_batch_prediction_feature_group_output(self.batch_prediction_id, table_name)

    def set_output_to_console(self):
        """
        Sets the batch prediction output to the console, clearing both the file connector and database connector config

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.set_batch_prediction_output_to_console(self.batch_prediction_id)

    def set_dataset(self, dataset_type: str, dataset_id: str = None):
        """
        [Deprecated] Sets the batch prediction input dataset. Only applicable for legacy dataset-based projects

        Args:
            dataset_type (str): The dataset type to set
            dataset_id (str): The dataset to set

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.set_batch_prediction_dataset(self.batch_prediction_id, dataset_type, dataset_id)

    def set_feature_group(self, feature_group_type: str, feature_group_id: str = None):
        """
        Sets the batch prediction input feature group.

        Args:
            feature_group_type (str): The feature group type to set. The feature group type of the feature group. The type is based on the use case under which the feature group is being created. For example, Catalog Attributes can be a feature group type under personalized recommendation use case.
            feature_group_id (str): The feature group to set as input to the batch prediction

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.set_batch_prediction_feature_group(self.batch_prediction_id, feature_group_type, feature_group_id)

    def set_dataset_remap(self, dataset_id_remap: dict):
        """
        For the purpose of this batch prediction, will swap out datasets in the input feature groups

        Args:
            dataset_id_remap (dict): Key/value pairs of dataset_ids to replace during batch predictions

        Returns:
            BatchPrediction: Batch Prediction object
        """
        return self.client.set_batch_prediction_dataset_remap(self.batch_prediction_id, dataset_id_remap)

    def delete(self):
        """
        Deletes a batch prediction and associated data such as associated monitors.

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction
        """
        return self.client.delete_batch_prediction(self.batch_prediction_id)

    def wait_for_predictions(self, timeout=86400):
        """
        A waiting call until batch predictions are ready.

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
        if self.describe().latest_batch_prediction_version:
            return self.describe().latest_batch_prediction_version.wait_for_drift_monitor(timeout=timeout)

    def get_status(self):
        """
        Gets the status of the latest batch prediction version.

        Returns:
            str: A string describing the status of the latest batch prediction version e.g., pending, complete, etc.
        """
        return self.describe().latest_batch_prediction_version.status

    def create_refresh_policy(self, cron: str):
        """
        To create a refresh policy for a batch prediction.

        Args:
            cron (str): A cron style string to set the refresh time.

        Returns:
            RefreshPolicy: The refresh policy object.
        """
        return self.client.create_refresh_policy(self.name, cron, 'BATCHPRED', batch_prediction_ids=[self.id])

    def list_refresh_policies(self):
        """
        Gets the refresh policies in a list.

        Returns:
            List[RefreshPolicy]: A list of refresh policy objects.
        """
        return self.client.list_refresh_policies(batch_prediction_ids=[self.id])

    def describe_output_feature_group(self):
        """
        Gets the results feature group for this batch prediction

        Returns:
            FeatureGroup: A feature group object.
        """
        if not self.feature_group_table_name:
            from .client import ApiException
            raise ApiException(
                'Batch prediction does not have a feature group output', 409, 'ConflictError')
        return self.client.describe_feature_group_by_table_name(self.feature_group_table_name)
