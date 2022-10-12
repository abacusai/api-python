from .feature_group_export_config import FeatureGroupExportConfig
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class Deployment(AbstractApiClass):
    """
        A model deployment

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentId (str): The unique identifier for the deployment.
            name (str): The user-friendly name for the deployment.
            status (str): The status of the deployment.
            description (str): A description of this deployment.
            deployedAt (str): When the deployment last became active.
            createdAt (str): When the deployment was created.
            projectId (str): The unique identifier of the project this deployment belongs to.
            modelId (str): The model that is currently deployed.
            modelVersion (str): The model version ID that is currently deployed.
            featureGroupId (str): The feature group that is currently deployed.
            featureGroupVersion (str): The feature group version ID that is currently deployed.
            callsPerSecond (int): The number of calls per second the deployment could handle.
            autoDeploy (bool): Flag marking the deployment eligible for auto deployments whenever any model in the project finishes training.
            algoName (str): 
            regions (list of strings): List of regions that a deployment has been deployed to
            error (str): Relevant error if the status is FAILED
            batchStreamingUpdates (bool): Flag marking the feature group deployment as having enabled a background process which caches streamed in rows for quicker lookup
            algorithm (str): The algorithm that is currently deployed.
            pendingModelVersion (dict): The model that the deployment is switching to.
            modelDeploymentConfig (dict): The config for what model to be deployed
            refreshSchedules (RefreshSchedule): List of refresh schedules that indicate when the deployment will be updated to the latest model version
            featureGroupExportConfig (FeatureGroupExportConfig): Export config (file connector or database connector information) for feature group deployment exports
    """

    def __init__(self, client, deploymentId=None, name=None, status=None, description=None, deployedAt=None, createdAt=None, projectId=None, modelId=None, modelVersion=None, featureGroupId=None, featureGroupVersion=None, callsPerSecond=None, autoDeploy=None, algoName=None, regions=None, error=None, batchStreamingUpdates=None, algorithm=None, pendingModelVersion=None, modelDeploymentConfig=None, refreshSchedules={}, featureGroupExportConfig={}):
        super().__init__(client, deploymentId)
        self.deployment_id = deploymentId
        self.name = name
        self.status = status
        self.description = description
        self.deployed_at = deployedAt
        self.created_at = createdAt
        self.project_id = projectId
        self.model_id = modelId
        self.model_version = modelVersion
        self.feature_group_id = featureGroupId
        self.feature_group_version = featureGroupVersion
        self.calls_per_second = callsPerSecond
        self.auto_deploy = autoDeploy
        self.algo_name = algoName
        self.regions = regions
        self.error = error
        self.batch_streaming_updates = batchStreamingUpdates
        self.algorithm = algorithm
        self.pending_model_version = pendingModelVersion
        self.model_deployment_config = modelDeploymentConfig
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.feature_group_export_config = client._build_class(
            FeatureGroupExportConfig, featureGroupExportConfig)

    def __repr__(self):
        return f"Deployment(deployment_id={repr(self.deployment_id)},\n  name={repr(self.name)},\n  status={repr(self.status)},\n  description={repr(self.description)},\n  deployed_at={repr(self.deployed_at)},\n  created_at={repr(self.created_at)},\n  project_id={repr(self.project_id)},\n  model_id={repr(self.model_id)},\n  model_version={repr(self.model_version)},\n  feature_group_id={repr(self.feature_group_id)},\n  feature_group_version={repr(self.feature_group_version)},\n  calls_per_second={repr(self.calls_per_second)},\n  auto_deploy={repr(self.auto_deploy)},\n  algo_name={repr(self.algo_name)},\n  regions={repr(self.regions)},\n  error={repr(self.error)},\n  batch_streaming_updates={repr(self.batch_streaming_updates)},\n  algorithm={repr(self.algorithm)},\n  pending_model_version={repr(self.pending_model_version)},\n  model_deployment_config={repr(self.model_deployment_config)},\n  refresh_schedules={repr(self.refresh_schedules)},\n  feature_group_export_config={repr(self.feature_group_export_config)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'deployment_id': self.deployment_id, 'name': self.name, 'status': self.status, 'description': self.description, 'deployed_at': self.deployed_at, 'created_at': self.created_at, 'project_id': self.project_id, 'model_id': self.model_id, 'model_version': self.model_version, 'feature_group_id': self.feature_group_id, 'feature_group_version': self.feature_group_version, 'calls_per_second': self.calls_per_second, 'auto_deploy': self.auto_deploy, 'algo_name': self.algo_name, 'regions': self.regions, 'error': self.error, 'batch_streaming_updates': self.batch_streaming_updates, 'algorithm': self.algorithm, 'pending_model_version': self.pending_model_version, 'model_deployment_config': self.model_deployment_config, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'feature_group_export_config': self._get_attribute_as_dict(self.feature_group_export_config)}

    def create_webhook(self, endpoint: str, webhook_event_type: str, payload_template: dict = None):
        """
        Create a webhook attached to a given deployment id.

        Args:
            endpoint (str): URI that the webhook will send HTTP POST requests to.
            webhook_event_type (str): One of 'DEPLOYMENT_START', 'DEPLOYMENT_SUCCESS', 'DEPLOYMENT_FAILED'
            payload_template (dict): Template for the body of the HTTP POST requests. Defaults to {}.

        Returns:
            Webhook: The Webhook attached to the deployment
        """
        return self.client.create_deployment_webhook(self.deployment_id, endpoint, webhook_event_type, payload_template)

    def list_webhooks(self):
        """
        List and describe all the webhooks attached to a given deployment ID.

        Args:
            deployment_id (str): ID of target deployment.

        Returns:
            Webhook: The webhooks attached to the given deployment id.
        """
        return self.client.list_deployment_webhooks(self.deployment_id)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            Deployment: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified deployment.

        Args:
            deployment_id (str): The unique ID associated with the deployment.

        Returns:
            Deployment: The description of the deployment.
        """
        return self.client.describe_deployment(self.deployment_id)

    def update(self, description: str = None):
        """
        Updates a deployment's description.

        Args:
            description (str): The new deployment description.
        """
        return self.client.update_deployment(self.deployment_id, description)

    def rename(self, name: str):
        """
        Updates a deployment's name and/or description.

        Args:
            name (str): The new deployment name.
        """
        return self.client.rename_deployment(self.deployment_id, name)

    def set_auto(self, enable: bool = None):
        """
        Enable/Disable auto deployment for the specified deployment.

        When a model is scheduled to retrain, deployments with this enabled will be marked to automatically promote the new model
        version. After the newly trained model completes, a check on its metrics in comparison to the currently deployed model version
        will be performed. If the metrics are comparable or better, the newly trained model version is automatically promoted. If not,
        it will be marked as a failed model version promotion with an error indicating poor metrics performance.


        Args:
            enable (bool): Enable/disable the autoDeploy property of the Deployment.
        """
        return self.client.set_auto_deployment(self.deployment_id, enable)

    def set_model_version(self, model_version: str, algorithm: str = None):
        """
        Promotes a Model Version to be served in the Deployment

        Args:
            model_version (str): The unique ID for the Model Version
            algorithm (str): 
        """
        return self.client.set_deployment_model_version(self.deployment_id, model_version, algorithm)

    def set_feature_group_version(self, feature_group_version: str):
        """
        Promotes a Feature Group Version to be served in the Deployment

        Args:
            feature_group_version (str): The unique ID for the Feature Group Version
        """
        return self.client.set_deployment_feature_group_version(self.deployment_id, feature_group_version)

    def start(self):
        """
        Restarts the specified deployment that was previously suspended.

        Args:
            deployment_id (str): The unique ID associated with the deployment.
        """
        return self.client.start_deployment(self.deployment_id)

    def stop(self):
        """
        Stops the specified deployment.

        Args:
            deployment_id (str): The Deployment ID
        """
        return self.client.stop_deployment(self.deployment_id)

    def delete(self):
        """
        Deletes the specified deployment. The deployment's models will not be affected. Note that the deployments are not recoverable after they are deleted.

        Args:
            deployment_id (str): The ID of the deployment to delete.
        """
        return self.client.delete_deployment(self.deployment_id)

    def set_feature_group_export_file_connector_output(self, file_format: str = None, output_location: str = None):
        """
        Sets the export output for the Feature Group Deployment to be a file connector.

        Args:
            file_format (str): 
            output_location (str): the file connector (cloud) location of where to export
        """
        return self.client.set_deployment_feature_group_export_file_connector_output(self.deployment_id, file_format, output_location)

    def set_feature_group_export_database_connector_output(self, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None):
        """
        Sets the export output for the Feature Group Deployment to be a Database connector.

        Args:
            database_connector_id (str): The database connector ID used
            object_name (str): The database connector's object to write to
            write_mode (str): UPSERT or INSERT for writing to the database connector
            database_feature_mapping (dict): The column/feature pairs mapping the features to the database columns
            id_column (str): The id column to use as the upsert key
            additional_id_columns (list): For database connectors which support it, additional ID columns to use as a complex key for upserting
        """
        return self.client.set_deployment_feature_group_export_database_connector_output(self.deployment_id, database_connector_id, object_name, write_mode, database_feature_mapping, id_column, additional_id_columns)

    def remove_feature_group_export_output(self):
        """
        Removes the export type that is set for the Feature Group Deployment

        Args:
            deployment_id (str): The deployment for which the export type is set
        """
        return self.client.remove_deployment_feature_group_export_output(self.deployment_id)

    def create_batch_prediction(self, table_name: str = None, name: str = None, global_prediction_args: dict = None, explanations: bool = False, output_format: str = None, output_location: str = None, database_connector_id: str = None, database_output_config: dict = None, refresh_schedule: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None):
        """
        Creates a batch prediction job description for the given deployment.

        Args:
            table_name (str): If specified, the name of the feature group table to write the results of the batch prediction. Can only be specified iff outputLocation and databaseConnectorId are not specified. If tableName is specified, the outputType will be enforced as CSV
            name (str): The name of batch prediction job.
            global_prediction_args (dict): Argument(s) to pass on every prediction call.
            explanations (bool): If true, will provide SHAP Explanations for each prediction, if supported by the use case.
            output_format (str): If specified, sets the format of the batch prediction output (CSV or JSON)
            output_location (str): If specified, the location to write the prediction results. Otherwise, results will be stored in Abacus.AI.
            database_connector_id (str): The unique identifier of an Database Connection to write predictions to. Cannot be specified in conjunction with outputLocation.
            database_output_config (dict): A key-value pair of columns/values to write to the database connector. Only available if databaseConnectorId is specified.
            refresh_schedule (str): A cron-style string that describes a schedule in UTC to automatically run the batch prediction.
            csv_input_prefix (str): A prefix to prepend to the input columns, only applies when output format is CSV
            csv_prediction_prefix (str): A prefix to prepend to the prediction columns, only applies when output format is CSV
            csv_explanations_prefix (str): A prefix to prepend to the explanation columns, only applies when output format is CSV
            output_includes_metadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.create_batch_prediction(self.deployment_id, table_name, name, global_prediction_args, explanations, output_format, output_location, database_connector_id, database_output_config, refresh_schedule, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix, output_includes_metadata, result_input_columns)

    def wait_for_deployment(self, wait_states={'PENDING', 'DEPLOYING'}, timeout=480):
        """
        A waiting call until deployment is completed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, wait_states, timeout=timeout)

    def wait_for_pending_deployment_update(self, timeout=600):
        """
        A waiting call until pending model switch is completed.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.

        Returns:
            Deployment: the latest deployment object.
        """
        import time
        start_time = time.time()
        while True:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            if not self.refresh().pending_model_version:
                break
            time.sleep(5)
        return self.refresh()

    def get_status(self):
        """
        Gets the status of the deployment.

        Returns:
            str: A string describing the status of a deploymet (pending, deploying, active, etc.).
        """
        return self.describe().status

    def create_refresh_policy(self, cron: str):
        """
        To create a refresh policy for a deployment.

        Args:
            cron (str): A cron style string to set the refresh time.

        Returns:
            RefreshPolicy: The refresh policy object.
        """
        return self.client.create_refresh_policy(self.name, cron, 'DEPLOYMENT', deployment_ids=[self.id])

    def list_refresh_policies(self):
        """
        Gets the refresh policies in a list.

        Returns:
            List[RefreshPolicy]: A list of refresh policy objects.
        """
        return self.client.list_refresh_policies(deployment_ids=[self.id])
