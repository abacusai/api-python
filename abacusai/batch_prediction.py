from .batch_prediction_version import BatchPredictionVersion
from .prediction_input import PredictionInput
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class BatchPrediction(AbstractApiClass):
    """
        Batch predictions
    """

    def __init__(self, client, batchPredictionId=None, createdAt=None, name=None, deploymentId=None, fileConnectorOutputLocation=None, globalPredictionArgs=None, databaseConnectorId=None, databaseOutputConfiguration=None, explanations=None, fileOutputFormat=None, connectorType=None, legacyInputLocation=None, featureGroupTableName=None, csvInputPrefix=None, csvPredictionPrefix=None, csvExplanationsPrefix=None, batchInputs={}, latestBatchPredictionVersion={}, refreshSchedules={}):
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
        self.csv_input_prefix = csvInputPrefix
        self.csv_prediction_prefix = csvPredictionPrefix
        self.csv_explanations_prefix = csvExplanationsPrefix
        self.batch_inputs = client._build_class(PredictionInput, batchInputs)
        self.latest_batch_prediction_version = client._build_class(
            BatchPredictionVersion, latestBatchPredictionVersion)
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)

    def __repr__(self):
        return f"BatchPrediction(batch_prediction_id={repr(self.batch_prediction_id)},\n  created_at={repr(self.created_at)},\n  name={repr(self.name)},\n  deployment_id={repr(self.deployment_id)},\n  file_connector_output_location={repr(self.file_connector_output_location)},\n  global_prediction_args={repr(self.global_prediction_args)},\n  database_connector_id={repr(self.database_connector_id)},\n  database_output_configuration={repr(self.database_output_configuration)},\n  explanations={repr(self.explanations)},\n  file_output_format={repr(self.file_output_format)},\n  connector_type={repr(self.connector_type)},\n  legacy_input_location={repr(self.legacy_input_location)},\n  feature_group_table_name={repr(self.feature_group_table_name)},\n  csv_input_prefix={repr(self.csv_input_prefix)},\n  csv_prediction_prefix={repr(self.csv_prediction_prefix)},\n  csv_explanations_prefix={repr(self.csv_explanations_prefix)},\n  batch_inputs={repr(self.batch_inputs)},\n  latest_batch_prediction_version={repr(self.latest_batch_prediction_version)},\n  refresh_schedules={repr(self.refresh_schedules)})"

    def to_dict(self):
        return {'batch_prediction_id': self.batch_prediction_id, 'created_at': self.created_at, 'name': self.name, 'deployment_id': self.deployment_id, 'file_connector_output_location': self.file_connector_output_location, 'global_prediction_args': self.global_prediction_args, 'database_connector_id': self.database_connector_id, 'database_output_configuration': self.database_output_configuration, 'explanations': self.explanations, 'file_output_format': self.file_output_format, 'connector_type': self.connector_type, 'legacy_input_location': self.legacy_input_location, 'feature_group_table_name': self.feature_group_table_name, 'csv_input_prefix': self.csv_input_prefix, 'csv_prediction_prefix': self.csv_prediction_prefix, 'csv_explanations_prefix': self.csv_explanations_prefix, 'batch_inputs': self._get_attribute_as_dict(self.batch_inputs), 'latest_batch_prediction_version': self._get_attribute_as_dict(self.latest_batch_prediction_version), 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules)}

    def start(self):
        """Creates a new batch prediction version job for a given batch prediction job description"""
        return self.client.start_batch_prediction(self.batch_prediction_id)

    def refresh(self):
        """Calls describe and refreshes the current object's fields"""
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """Describes the batch prediction"""
        return self.client.describe_batch_prediction(self.batch_prediction_id)

    def list_versions(self, limit=100, start_after_version=None):
        """Retrieves a list of versions of a given batch prediction"""
        return self.client.list_batch_prediction_versions(self.batch_prediction_id, limit, start_after_version)

    def update(self, deployment_id=None, global_prediction_args=None, explanations=None, output_format=None, csv_input_prefix=None, csv_prediction_prefix=None, csv_explanations_prefix=None):
        """Updates a batch prediction job description"""
        return self.client.update_batch_prediction(self.batch_prediction_id, deployment_id, global_prediction_args, explanations, output_format, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix)

    def set_file_connector_output(self, output_format=None, output_location=None):
        """Updates the file connector output configuration of the batch prediction"""
        return self.client.set_batch_prediction_file_connector_output(self.batch_prediction_id, output_format, output_location)

    def set_database_connector_output(self, database_connector_id=None, database_output_config=None):
        """Updates the database connector output configuration of the batch prediction"""
        return self.client.set_batch_prediction_database_connector_output(self.batch_prediction_id, database_connector_id, database_output_config)

    def set_feature_group_output(self, table_name):
        """Creates a feature group and sets it to be the batch prediction output"""
        return self.client.set_batch_prediction_feature_group_output(self.batch_prediction_id, table_name)

    def set_output_to_console(self):
        """Sets the batch prediction output to the console, clearing both the file connector and database connector config"""
        return self.client.set_batch_prediction_output_to_console(self.batch_prediction_id)

    def set_dataset(self, dataset_type, dataset_id=None):
        """[Deprecated] Sets the batch prediction input dataset. Only applicable for legacy dataset-based projects"""
        return self.client.set_batch_prediction_dataset(self.batch_prediction_id, dataset_type, dataset_id)

    def set_feature_group(self, feature_group_type, feature_group_id=None):
        """Sets the batch prediction input feature group."""
        return self.client.set_batch_prediction_feature_group(self.batch_prediction_id, feature_group_type, feature_group_id)

    def set_dataset_remap(self, dataset_id_remap):
        """For the purpose of this batch prediction, will swap out datasets in the input feature groups"""
        return self.client.set_batch_prediction_dataset_remap(self.batch_prediction_id, dataset_id_remap)

    def delete(self):
        """Deletes a batch prediction"""
        return self.client.delete_batch_prediction(self.batch_prediction_id)

    def wait_for_predictions(self, timeout=86400):
        """
        A waiting call until batch predictions are ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out. Default value given is 86400 milliseconds.

        Returns:
            None
        """
        return self.client._poll(self, {'PENDING', 'UPLOADING', 'PREDICTING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the latest batch prediction version.

        Returns:
            Enum (string): A string describing the status of the latest batch prediction version e.g., pending, complete, etc.
        """
        return self.describe().latest_batch_prediction_version.status

    def create_refresh_policy(self, cron: str):
        """
        To create a refresh policy for a batch prediction.

        Args:
            cron (str): A cron style string to set the refresh time.

        Returns:
            RefreshPolicy (object): The refresh policy object.
        """
        return self.client.create_refresh_policy(self.name, cron, 'BATCHPRED', batch_prediction_ids=[self.id])

    def list_refresh_policies(self):
        """
        Gets the refresh policies in a list.

        Returns:
            List (RefreshPolicy): A list of refresh policy objects.
        """
        return self.client.list_refresh_policies(batch_prediction_ids=[self.id])
