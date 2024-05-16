from typing import Union

from . import api_class
from .api_class import BatchPredictionArgs
from .batch_prediction_version import BatchPredictionVersion
from .prediction_feature_group import PredictionFeatureGroup
from .prediction_input import PredictionInput
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class BatchPrediction(AbstractApiClass):
    """
        Make batch predictions.

        Args:
            client (ApiClient): An authenticated API Client instance
            batchPredictionId (str): The unique identifier of the batch prediction request.
            createdAt (str): When the batch prediction was created, in ISO-8601 format.
            name (str): Name given to the batch prediction object.
            deploymentId (str): The deployment used to make the predictions.
            fileConnectorOutputLocation (str): Contains information about where the batch predictions are written to.
            databaseConnectorId (str): The database connector to write the results to.
            databaseOutputConfiguration (dict): Contains information about where the batch predictions are written to.
            fileOutputFormat (str): The format of the batch prediction output (CSV or JSON).
            connectorType (str): Null if writing to internal console, else FEATURE_GROUP | FILE_CONNECTOR | DATABASE_CONNECTOR.
            legacyInputLocation (str): The location of the input data.
            outputFeatureGroupId (str): The Batch Prediction output feature group ID if applicable
            featureGroupTableName (str): The table name of the Batch Prediction output feature group.
            outputFeatureGroupTableName (str): The table name of the Batch Prediction output feature group.
            summaryFeatureGroupTableName (str): The table name of the metrics summary feature group output by Batch Prediction.
            csvInputPrefix (str): A prefix to prepend to the input columns, only applies when output format is CSV.
            csvPredictionPrefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV.
            csvExplanationsPrefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV.
            outputIncludesMetadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version.
            resultInputColumns (list): If present, will limit result files or feature groups to only include columns present in this list.
            modelMonitorId (str): The model monitor for this batch prediction.
            modelVersion (str): The model instance used in the deployment for the batch prediction.
            bpAcrossVersionsMonitorId (str): The model monitor for this batch prediction across versions.
            algorithm (str): The algorithm that is currently deployed.
            batchPredictionArgsType (str): The type of batch prediction arguments used for this batch prediction.
            batchInputs (PredictionInput): Inputs to the batch prediction.
            latestBatchPredictionVersion (BatchPredictionVersion): The latest batch prediction version.
            refreshSchedules (RefreshSchedule): List of refresh schedules that dictate the next time the batch prediction will be run.
            inputFeatureGroups (PredictionFeatureGroup): List of prediction feature groups.
            globalPredictionArgs (BatchPredictionArgs): 
            batchPredictionArgs (BatchPredictionArgs): Argument(s) passed to every prediction call.
    """

    def __init__(self, client, batchPredictionId=None, createdAt=None, name=None, deploymentId=None, fileConnectorOutputLocation=None, databaseConnectorId=None, databaseOutputConfiguration=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, outputFeatureGroupId=None, featureGroupTableName=None, outputFeatureGroupTableName=None, summaryFeatureGroupTableName=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, outputIncludesMetadata=None, resultInputColumns=None, modelMonitorId=None, modelVersion=None, bpAcrossVersionsMonitorId=None, algorithm=None, batchPredictionArgsType=None, batchInputs={}, latestBatchPredictionVersion={}, refreshSchedules={}, inputFeatureGroups={}, globalPredictionArgs={}, batchPredictionArgs={}):
        super().__init__(client, batchPredictionId)
        self.batch_prediction_id = batchPredictionId
        self.created_at = createdAt
        self.name = name
        self.deployment_id = deploymentId
        self.file_connector_output_location = fileConnectorOutputLocation
        self.database_connector_id = databaseConnectorId
        self.database_output_configuration = databaseOutputConfiguration
        self.file_output_format = fileOutputFormat
        self.connector_type = connectorType
        self.legacy_input_location = legacyInputLocation
        self.output_feature_group_id = outputFeatureGroupId
        self.feature_group_table_name = featureGroupTableName
        self.output_feature_group_table_name = outputFeatureGroupTableName
        self.summary_feature_group_table_name = summaryFeatureGroupTableName
        self.csv_input_prefix = csvInputPrefix
        self.csv_prediction_prefix = csvPredictionPrefix
        self.csv_explanations_prefix = csvExplanationsPrefix
        self.output_includes_metadata = outputIncludesMetadata
        self.result_input_columns = resultInputColumns
        self.model_monitor_id = modelMonitorId
        self.model_version = modelVersion
        self.bp_across_versions_monitor_id = bpAcrossVersionsMonitorId
        self.algorithm = algorithm
        self.batch_prediction_args_type = batchPredictionArgsType
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)
        self.latest_batch_prediction_version = client._build_class(
            BatchPredictionVersion, latestBatchPredictionVersion)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.input_feature_groups = client._build_class(
            PredictionFeatureGroup, inputFeatureGroups)
        self.global_prediction_args = client._build_class(
            BatchPredictionArgs, globalPredictionArgs)
        self.batch_prediction_args = client._build_class(getattr(
            api_class, batchPredictionArgsType, BatchPredictionArgs) if batchPredictionArgsType else BatchPredictionArgs, batchPredictionArgs)
        self.deprecated_keys = {'explanations', 'global_prediction_args'}

    def __repr__(self):
        repr_dict = {f'batch_prediction_id': repr(self.batch_prediction_id), f'created_at': repr(self.created_at), f'name': repr(self.name), f'deployment_id': repr(self.deployment_id), f'file_connector_output_location': repr(self.file_connector_output_location), f'database_connector_id': repr(self.database_connector_id), f'database_output_configuration': repr(self.database_output_configuration), f'file_output_format': repr(self.file_output_format), f'connector_type': repr(self.connector_type), f'legacy_input_location': repr(self.legacy_input_location), f'output_feature_group_id': repr(self.output_feature_group_id), f'feature_group_table_name': repr(self.feature_group_table_name), f'output_feature_group_table_name': repr(self.output_feature_group_table_name), f'summary_feature_group_table_name': repr(self.summary_feature_group_table_name), f'csv_input_prefix': repr(
            self.csv_input_prefix), f'csv_prediction_prefix': repr(self.csv_prediction_prefix), f'csv_explanations_prefix': repr(self.csv_explanations_prefix), f'output_includes_metadata': repr(self.output_includes_metadata), f'result_input_columns': repr(self.result_input_columns), f'model_monitor_id': repr(self.model_monitor_id), f'model_version': repr(self.model_version), f'bp_across_versions_monitor_id': repr(self.bp_across_versions_monitor_id), f'algorithm': repr(self.algorithm), f'batch_prediction_args_type': repr(self.batch_prediction_args_type), f'batch_inputs': repr(self.batch_inputs), f'latest_batch_prediction_version': repr(self.latest_batch_prediction_version), f'refresh_schedules': repr(self.refresh_schedules), f'input_feature_groups': repr(self.input_feature_groups), f'global_prediction_args': repr(self.global_prediction_args), f'batch_prediction_args': repr(self.batch_prediction_args)}
        class_name = "BatchPrediction"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'batch_prediction_id': self.batch_prediction_id, 'created_at': self.created_at, 'name': self.name, 'deployment_id': self.deployment_id, 'file_connector_output_location': self.file_connector_output_location, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'output_feature_group_id': self.output_feature_group_id, 'feature_group_table_name': self.feature_group_table_name, 'output_feature_group_table_name': self.output_feature_group_table_name, 'summary_feature_group_table_name': self.summary_feature_group_table_name, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix':
                self.csv_explanations_prefix, 'output_includes_metadata': self.output_includes_metadata, 'result_input_columns': self.result_input_columns, 'model_monitor_id': self.model_monitor_id, 'model_version': self.model_version, 'bp_across_versions_monitor_id': self.bp_across_versions_monitor_id, 'algorithm': self.algorithm, 'batch_prediction_args_type': self.batch_prediction_args_type, 'batch_inputs': self._get_attribute_as_dict(self.batch_inputs), 'latest_batch_prediction_version': self._get_attribute_as_dict(self.latest_batch_prediction_version), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'input_feature_groups': self._get_attribute_as_dict(self.input_feature_groups), 'global_prediction_args': self._get_attribute_as_dict(self.global_prediction_args), 'batch_prediction_args': self._get_attribute_as_dict(self.batch_prediction_args)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def start(self):
        """
        Creates a new batch prediction version job for a given batch prediction job description.

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction to create a new version of.

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
        Describe the batch prediction.

        Args:
            batch_prediction_id (str): The unique identifier associated with the batch prediction.

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.describe_batch_prediction(self.batch_prediction_id)

    def list_versions(self, limit: int = 100, start_after_version: str = None):
        """
        Retrieves a list of versions of a given batch prediction

        Args:
            limit (int): Number of versions to list.
            start_after_version (str): Version to start after.

        Returns:
            list[BatchPredictionVersion]: List of batch prediction versions.
        """
        return self.client.list_batch_prediction_versions(self.batch_prediction_id, limit, start_after_version)

    def update(self, deployment_id: str = None, global_prediction_args: Union[dict, BatchPredictionArgs] = None, batch_prediction_args: Union[dict, BatchPredictionArgs] = None, explanations: bool = None, output_format: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None, name: str = None):
        """
        Update a batch prediction job description.

        Args:
            deployment_id (str): Unique identifier of the deployment.
            batch_prediction_args (BatchPredictionArgs): Batch Prediction args specific to problem type.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON).
            csv_input_prefix (str): Prefix to prepend to the input columns, only applies when output format is CSV.
            csv_prediction_prefix (str): Prefix to prepend to the prediction columns, only applies when output format is CSV.
            csv_explanations_prefix (str): Prefix to prepend to the explanation columns, only applies when output format is CSV.
            output_includes_metadata (bool): If True, output will contain columns including prediction start time, batch prediction version, and model version.
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list.
            name (str): If present, will rename the batch prediction.

        Returns:
            BatchPrediction: The batch prediction.
        """
        return self.client.update_batch_prediction(self.batch_prediction_id, deployment_id, global_prediction_args, batch_prediction_args, explanations, output_format, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix, output_includes_metadata, result_input_columns, name)

    def set_file_connector_output(self, output_format: str = None, output_location: str = None):
        """
        Updates the file connector output configuration of the batch prediction

        Args:
            output_format (str): The format of the batch prediction output (CSV or JSON). If not specified, the default format will be used.
            output_location (str): The location to write the prediction results. If not specified, results will be stored in Abacus.AI.

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.set_batch_prediction_file_connector_output(self.batch_prediction_id, output_format, output_location)

    def set_database_connector_output(self, database_connector_id: str = None, database_output_config: dict = None):
        """
        Updates the database connector output configuration of the batch prediction

        Args:
            database_connector_id (str): Unique string identifier of an Database Connection to write predictions to.
            database_output_config (dict): Key-value pair of columns/values to write to the database connector.

        Returns:
            BatchPrediction: Description of the batch prediction.
        """
        return self.client.set_batch_prediction_database_connector_output(self.batch_prediction_id, database_connector_id, database_output_config)

    def set_feature_group_output(self, table_name: str):
        """
        Creates a feature group and sets it as the batch prediction output.

        Args:
            table_name (str): Name of the feature group table to create.

        Returns:
            BatchPrediction: Batch prediction after the output has been applied.
        """
        return self.client.set_batch_prediction_feature_group_output(self.batch_prediction_id, table_name)

    def set_output_to_console(self):
        """
        Sets the batch prediction output to the console, clearing both the file connector and database connector configurations.

        Args:
            batch_prediction_id (str): The unique identifier of the batch prediction.

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.set_batch_prediction_output_to_console(self.batch_prediction_id)

    def set_feature_group(self, feature_group_type: str, feature_group_id: str = None):
        """
        Sets the batch prediction input feature group.

        Args:
            feature_group_type (str): Enum string representing the feature group type to set. The type is based on the use case under which the feature group is being created (e.g. Catalog Attributes for personalized recommendation use case).
            feature_group_id (str): Unique identifier of the feature group to set as input to the batch prediction.

        Returns:
            BatchPrediction: Description of the batch prediction.
        """
        return self.client.set_batch_prediction_feature_group(self.batch_prediction_id, feature_group_type, feature_group_id)

    def set_dataset_remap(self, dataset_id_remap: dict):
        """
        For the purpose of this batch prediction, will swap out datasets in the training feature groups

        Args:
            dataset_id_remap (dict): Key/value pairs of dataset ids to be replaced during the batch prediction.

        Returns:
            BatchPrediction: Batch prediction object.
        """
        return self.client.set_batch_prediction_dataset_remap(self.batch_prediction_id, dataset_id_remap)

    def delete(self):
        """
        Deletes a batch prediction and associated data, such as associated monitors.

        Args:
            batch_prediction_id (str): Unique string identifier of the batch prediction.
        """
        return self.client.delete_batch_prediction(self.batch_prediction_id)

    def wait_for_predictions(self, timeout=86400):
        """
        A waiting call until batch predictions are ready.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def wait_for_drift_monitor(self, timeout=86400):
        """
        A waiting call until batch prediction drift monitor calculations are ready.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
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

    def load_results_as_pandas(self):
        """
        Loads the output feature groups into a python pandas dataframe.

        Returns:
            DataFrame: A pandas dataframe with annotations and text_snippet columns.
        """
        return self.describe().latest_batch_prediction_version.wait_for_predictions().load_results_as_pandas()
