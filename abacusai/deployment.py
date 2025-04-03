from typing import Union

from .api_class import (
    AlertActionConfig, AlertConditionConfig, BatchPredictionArgs,
    DeploymentConversationType, FeatureGroupExportConfig, PredictionArguments
)
from .feature_group_export_config import FeatureGroupExportConfig
from .refresh_schedule import RefreshSchedule
from .return_class import AbstractApiClass


class Deployment(AbstractApiClass):
    """
        A model deployment

        Args:
            client (ApiClient): An authenticated API Client instance
            deploymentId (str): A unique identifier for the deployment.
            name (str): A user-friendly name for the deployment.
            status (str): The status of the deployment.
            description (str): A description of the deployment.
            deployedAt (str): The date and time when the deployment became active, in ISO-8601 format.
            createdAt (str): The date and time when the deployment was created, in ISO-8601 format.
            projectId (str): A unique identifier for the project this deployment belongs to.
            modelId (str): The model that is currently deployed.
            modelVersion (str): The model version ID that is currently deployed.
            featureGroupId (str): The feature group that is currently deployed.
            featureGroupVersion (str): The feature group version ID that is currently deployed.
            callsPerSecond (int): The number of calls per second the deployment can handle.
            autoDeploy (bool): A flag marking the deployment as eligible for auto deployments whenever any model in the project finishes training.
            skipMetricsCheck (bool): A flag to skip metric regression with this current deployment. This field is only relevant when auto_deploy is on
            algoName (str): The name of the algorithm that is currently deployed.
            regions (list): A list of regions that the deployment has been deployed to.
            error (str): The relevant error, if the status is FAILED.
            batchStreamingUpdates (bool): A flag marking the feature group deployment as having enabled a background process which caches streamed-in rows for quicker lookup.
            algorithm (str): The algorithm that is currently deployed.
            pendingModelVersion (dict): The model that the deployment is switching to, or being stopped.
            modelDeploymentConfig (dict): The config for which model to be deployed.
            predictionOperatorId (str): The prediction operator ID that is currently deployed.
            predictionOperatorVersion (str): The prediction operator version ID that is currently deployed.
            pendingPredictionOperatorVersion (str): The prediction operator version ID that the deployment is switching to, or being stopped.
            onlineFeatureGroupId (id): The online feature group ID that the deployment is running on
            outputOnlineFeatureGroupId (id): The online feature group ID that the deployment is outputting results to
            realtimeMonitorId (id): The realtime monitor ID of the realtime-monitor that is associated with the deployment
            runtimeConfigs (dict): The runtime configurations of a deployment which is used by some of the usecases during prediction.
            isSystemCreated (bool): Whether the deployment is system created.
            refreshSchedules (RefreshSchedule): A list of refresh schedules that indicate when the deployment will be updated to the latest model version.
            featureGroupExportConfig (FeatureGroupExportConfig): The export config (file connector or database connector information) for feature group deployment exports.
            defaultPredictionArguments (PredictionArguments): The default prediction arguments for prediction APIs
    """

    def __init__(self, client, deploymentId=None, name=None, status=None, description=None, deployedAt=None, createdAt=None, projectId=None, modelId=None, modelVersion=None, featureGroupId=None, featureGroupVersion=None, callsPerSecond=None, autoDeploy=None, skipMetricsCheck=None, algoName=None, regions=None, error=None, batchStreamingUpdates=None, algorithm=None, pendingModelVersion=None, modelDeploymentConfig=None, predictionOperatorId=None, predictionOperatorVersion=None, pendingPredictionOperatorVersion=None, onlineFeatureGroupId=None, outputOnlineFeatureGroupId=None, realtimeMonitorId=None, runtimeConfigs=None, isSystemCreated=None, refreshSchedules={}, featureGroupExportConfig={}, defaultPredictionArguments={}):
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
        self.skip_metrics_check = skipMetricsCheck
        self.algo_name = algoName
        self.regions = regions
        self.error = error
        self.batch_streaming_updates = batchStreamingUpdates
        self.algorithm = algorithm
        self.pending_model_version = pendingModelVersion
        self.model_deployment_config = modelDeploymentConfig
        self.prediction_operator_id = predictionOperatorId
        self.prediction_operator_version = predictionOperatorVersion
        self.pending_prediction_operator_version = pendingPredictionOperatorVersion
        self.online_feature_group_id = onlineFeatureGroupId
        self.output_online_feature_group_id = outputOnlineFeatureGroupId
        self.realtime_monitor_id = realtimeMonitorId
        self.runtime_configs = runtimeConfigs
        self.is_system_created = isSystemCreated
        self.refresh_schedules = client._build_class(
            RefreshSchedule, refreshSchedules)
        self.feature_group_export_config = client._build_class(
            FeatureGroupExportConfig, featureGroupExportConfig)
        self.default_prediction_arguments = client._build_class(
            PredictionArguments, defaultPredictionArguments)
        self.deprecated_keys = {}

    def __repr__(self):
        repr_dict = {f'deployment_id': repr(self.deployment_id), f'name': repr(self.name), f'status': repr(self.status), f'description': repr(self.description), f'deployed_at': repr(self.deployed_at), f'created_at': repr(self.created_at), f'project_id': repr(self.project_id), f'model_id': repr(self.model_id), f'model_version': repr(self.model_version), f'feature_group_id': repr(self.feature_group_id), f'feature_group_version': repr(self.feature_group_version), f'calls_per_second': repr(self.calls_per_second), f'auto_deploy': repr(self.auto_deploy), f'skip_metrics_check': repr(self.skip_metrics_check), f'algo_name': repr(self.algo_name), f'regions': repr(self.regions), f'error': repr(self.error), f'batch_streaming_updates': repr(self.batch_streaming_updates), f'algorithm': repr(self.algorithm), f'pending_model_version': repr(
            self.pending_model_version), f'model_deployment_config': repr(self.model_deployment_config), f'prediction_operator_id': repr(self.prediction_operator_id), f'prediction_operator_version': repr(self.prediction_operator_version), f'pending_prediction_operator_version': repr(self.pending_prediction_operator_version), f'online_feature_group_id': repr(self.online_feature_group_id), f'output_online_feature_group_id': repr(self.output_online_feature_group_id), f'realtime_monitor_id': repr(self.realtime_monitor_id), f'runtime_configs': repr(self.runtime_configs), f'is_system_created': repr(self.is_system_created), f'refresh_schedules': repr(self.refresh_schedules), f'feature_group_export_config': repr(self.feature_group_export_config), f'default_prediction_arguments': repr(self.default_prediction_arguments)}
        class_name = "Deployment"
        repr_str = ',\n  '.join([f'{key}={value}' for key, value in repr_dict.items(
        ) if getattr(self, key, None) is not None and key not in self.deprecated_keys])
        return f"{class_name}({repr_str})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        resp = {'deployment_id': self.deployment_id, 'name': self.name, 'status': self.status, 'description': self.description, 'deployed_at': self.deployed_at, 'created_at': self.created_at, 'project_id': self.project_id, 'model_id': self.model_id, 'model_version': self.model_version, 'feature_group_id': self.feature_group_id, 'feature_group_version': self.feature_group_version, 'calls_per_second': self.calls_per_second, 'auto_deploy': self.auto_deploy, 'skip_metrics_check': self.skip_metrics_check, 'algo_name': self.algo_name, 'regions': self.regions, 'error': self.error, 'batch_streaming_updates': self.batch_streaming_updates, 'algorithm': self.algorithm, 'pending_model_version': self.pending_model_version, 'model_deployment_config': self.model_deployment_config,
                'prediction_operator_id': self.prediction_operator_id, 'prediction_operator_version': self.prediction_operator_version, 'pending_prediction_operator_version': self.pending_prediction_operator_version, 'online_feature_group_id': self.online_feature_group_id, 'output_online_feature_group_id': self.output_online_feature_group_id, 'realtime_monitor_id': self.realtime_monitor_id, 'runtime_configs': self.runtime_configs, 'is_system_created': self.is_system_created, 'refresh_schedules': self._get_attribute_as_dict(self.refresh_schedules), 'feature_group_export_config': self._get_attribute_as_dict(self.feature_group_export_config), 'default_prediction_arguments': self._get_attribute_as_dict(self.default_prediction_arguments)}
        return {key: value for key, value in resp.items() if value is not None and key not in self.deprecated_keys}

    def create_webhook(self, endpoint: str, webhook_event_type: str, payload_template: dict = None):
        """
        Create a webhook attached to a given deployment ID.

        Args:
            endpoint (str): URI that the webhook will send HTTP POST requests to.
            webhook_event_type (str): One of 'DEPLOYMENT_START', 'DEPLOYMENT_SUCCESS', or 'DEPLOYMENT_FAILED'.
            payload_template (dict): Template for the body of the HTTP POST requests. Defaults to {}.

        Returns:
            Webhook: The webhook attached to the deployment.
        """
        return self.client.create_deployment_webhook(self.deployment_id, endpoint, webhook_event_type, payload_template)

    def list_webhooks(self):
        """
        List all the webhooks attached to a given deployment.

        Args:
            deployment_id (str): Unique identifier of the target deployment.

        Returns:
            list[Webhook]: List of the webhooks attached to the given deployment ID.
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
            deployment_id (str): Unique string identifier associated with the deployment.

        Returns:
            Deployment: Description of the deployment.
        """
        return self.client.describe_deployment(self.deployment_id)

    def update(self, description: str = None, auto_deploy: bool = None, skip_metrics_check: bool = None):
        """
        Updates a deployment's properties.

        Args:
            description (str): The new description for the deployment.
            auto_deploy (bool): Flag to enable the automatic deployment when a new Model Version finishes training.
            skip_metrics_check (bool): Flag to skip metric regression with this current deployment. This field is only relevant when auto_deploy is on
        """
        return self.client.update_deployment(self.deployment_id, description, auto_deploy, skip_metrics_check)

    def rename(self, name: str):
        """
        Updates a deployment's name

        Args:
            name (str): The new deployment name.
        """
        return self.client.rename_deployment(self.deployment_id, name)

    def set_auto(self, enable: bool = None):
        """
        Enable or disable auto deployment for the specified deployment.

        When a model is scheduled to retrain, deployments with auto deployment enabled will be marked to automatically promote the new model version. After the newly trained model completes, a check on its metrics in comparison to the currently deployed model version will be performed. If the metrics are comparable or better, the newly trained model version is automatically promoted. If not, it will be marked as a failed model version promotion with an error indicating poor metrics performance.


        Args:
            enable (bool): Enable or disable the autoDeploy property of the deployment.
        """
        return self.client.set_auto_deployment(self.deployment_id, enable)

    def set_model_version(self, model_version: str, algorithm: str = None, model_deployment_config: dict = None):
        """
        Promotes a model version and/or algorithm to be the active served deployment version

        Args:
            model_version (str): A unique identifier for the model version.
            algorithm (str): The algorithm to use for the model version. If not specified, the algorithm will be inferred from the model version.
            model_deployment_config (dict): The deployment configuration for the model to deploy.
        """
        return self.client.set_deployment_model_version(self.deployment_id, model_version, algorithm, model_deployment_config)

    def set_feature_group_version(self, feature_group_version: str):
        """
        Promotes a feature group version to be served in the deployment.

        Args:
            feature_group_version (str): Unique string identifier for the feature group version.
        """
        return self.client.set_deployment_feature_group_version(self.deployment_id, feature_group_version)

    def set_prediction_operator_version(self, prediction_operator_version: str):
        """
        Promotes a prediction operator version to be served in the deployment.

        Args:
            prediction_operator_version (str): Unique string identifier for the prediction operator version.
        """
        return self.client.set_deployment_prediction_operator_version(self.deployment_id, prediction_operator_version)

    def start(self):
        """
        Restarts the specified deployment that was previously suspended.

        Args:
            deployment_id (str): A unique string identifier associated with the deployment.
        """
        return self.client.start_deployment(self.deployment_id)

    def stop(self):
        """
        Stops the specified deployment.

        Args:
            deployment_id (str): Unique string identifier of the deployment to be stopped.
        """
        return self.client.stop_deployment(self.deployment_id)

    def delete(self):
        """
        Deletes the specified deployment. The deployment's models will not be affected. Note that the deployments are not recoverable after they are deleted.

        Args:
            deployment_id (str): Unique string identifier of the deployment to delete.
        """
        return self.client.delete_deployment(self.deployment_id)

    def set_feature_group_export_file_connector_output(self, file_format: str = None, output_location: str = None):
        """
        Sets the export output for the Feature Group Deployment to be a file connector.

        Args:
            file_format (str): The type of export output, either CSV or JSON.
            output_location (str): The file connector (cloud) location where the output should be exported.
        """
        return self.client.set_deployment_feature_group_export_file_connector_output(self.deployment_id, file_format, output_location)

    def set_feature_group_export_database_connector_output(self, database_connector_id: str, object_name: str, write_mode: str, database_feature_mapping: dict, id_column: str = None, additional_id_columns: list = None):
        """
        Sets the export output for the Feature Group Deployment to a Database connector.

        Args:
            database_connector_id (str): The unique string identifier of the database connector used.
            object_name (str): The object of the database connector to write to.
            write_mode (str): The write mode to use when writing to the database connector, either UPSERT or INSERT.
            database_feature_mapping (dict): The column/feature pairs mapping the features to the database columns.
            id_column (str): The id column to use as the upsert key.
            additional_id_columns (list): For database connectors which support it, a list of additional ID columns to use as a complex key for upserting.
        """
        return self.client.set_deployment_feature_group_export_database_connector_output(self.deployment_id, database_connector_id, object_name, write_mode, database_feature_mapping, id_column, additional_id_columns)

    def remove_feature_group_export_output(self):
        """
        Removes the export type that is set for the Feature Group Deployment

        Args:
            deployment_id (str): The ID of the deployment for which the export type is set.
        """
        return self.client.remove_deployment_feature_group_export_output(self.deployment_id)

    def set_default_prediction_arguments(self, prediction_arguments: Union[dict, PredictionArguments], set_as_override: bool = False):
        """
        Sets the deployment config.

        Args:
            prediction_arguments (PredictionArguments): The prediction arguments to set.
            set_as_override (bool): If True, use these arguments as overrides instead of defaults for predict calls

        Returns:
            Deployment: description of the updated deployment.
        """
        return self.client.set_default_prediction_arguments(self.deployment_id, prediction_arguments, set_as_override)

    def get_prediction_logs_records(self, limit: int = 10, last_log_request_id: str = '', last_log_timestamp: int = None):
        """
        Retrieves the prediction request IDs for the most recent predictions made to the deployment.

        Args:
            limit (int): The number of prediction log entries to retrieve up to the specified limit.
            last_log_request_id (str): The request ID of the last log entry to retrieve.
            last_log_timestamp (int): A Unix timestamp in milliseconds specifying the timestamp for the last log entry.

        Returns:
            list[PredictionLogRecord]: A list of prediction log records.
        """
        return self.client.get_prediction_logs_records(self.deployment_id, limit, last_log_request_id, last_log_timestamp)

    def create_alert(self, alert_name: str, condition_config: Union[dict, AlertConditionConfig], action_config: Union[dict, AlertActionConfig]):
        """
        Create a deployment alert for the given conditions.

        Only support batch prediction usage now.


        Args:
            alert_name (str): Name of the alert.
            condition_config (AlertConditionConfig): Condition to run the actions for the alert.
            action_config (AlertActionConfig): Configuration for the action of the alert.

        Returns:
            MonitorAlert: Object describing the deployment alert.
        """
        return self.client.create_deployment_alert(self.deployment_id, alert_name, condition_config, action_config)

    def list_alerts(self):
        """
        List the monitor alerts associated with the deployment id.

        Args:
            deployment_id (str): Unique string identifier for the deployment.

        Returns:
            list[MonitorAlert]: An array of deployment alerts.
        """
        return self.client.list_deployment_alerts(self.deployment_id)

    def create_realtime_monitor(self, realtime_monitor_schedule: str = None, lookback_time: int = None):
        """
        Real time monitors compute and monitor metrics of real time prediction data.

        Args:
            realtime_monitor_schedule (str): The cron expression for triggering monitor.
            lookback_time (int): Lookback time (in seconds) for each monitor trigger

        Returns:
            RealtimeMonitor: Object describing the real-time monitor.
        """
        return self.client.create_realtime_monitor(self.deployment_id, realtime_monitor_schedule, lookback_time)

    def get_conversation_response(self, message: str, deployment_token: str, deployment_conversation_id: str = None, external_session_id: str = None, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, doc_infos: list = None):
        """
        Return a conversation response which continues the conversation based on the input message and deployment conversation id (if exists).

        Args:
            message (str): A message from the user
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifiying the query chat config override.
            doc_infos (list): An optional list of documents use for the conversation. A keyword 'doc_id' is expected to be present in each document for retrieving contents from docstore.
        """
        return self.client.get_conversation_response(self.deployment_id, message, deployment_token, deployment_conversation_id, external_session_id, llm_name, num_completion_tokens, system_message, temperature, filter_key_values, search_score_cutoff, chat_config, doc_infos)

    def get_conversation_response_with_binary_data(self, deployment_token: str, message: str, deployment_conversation_id: str = None, external_session_id: str = None, llm_name: str = None, num_completion_tokens: int = None, system_message: str = None, temperature: float = 0.0, filter_key_values: dict = None, search_score_cutoff: float = None, chat_config: dict = None, attachments: None = None):
        """
        Return a conversation response which continues the conversation based on the input message and deployment conversation id (if exists).

        Args:
            deployment_token (str): A token used to authenticate access to deployments created in this project. This token is only authorized to predict on deployments in this project, so it is safe to embed this model inside of an application or website.
            message (str): A message from the user
            deployment_conversation_id (str): The unique identifier of a deployment conversation to continue. If not specified, a new one will be created.
            external_session_id (str): The user supplied unique identifier of a deployment conversation to continue. If specified, we will use this instead of a internal deployment conversation id.
            llm_name (str): Name of the specific LLM backend to use to power the chat experience
            num_completion_tokens (int): Default for maximum number of tokens for chat answers
            system_message (str): The generative LLM system message
            temperature (float): The generative LLM temperature
            filter_key_values (dict): A dictionary mapping column names to a list of values to restrict the retrived search results.
            search_score_cutoff (float): Cutoff for the document retriever score. Matching search results below this score will be ignored.
            chat_config (dict): A dictionary specifiying the query chat config override.
            attachments (None): A dictionary of binary data to use to answer the queries.
        """
        return self.client.get_conversation_response_with_binary_data(self.deployment_id, deployment_token, message, deployment_conversation_id, external_session_id, llm_name, num_completion_tokens, system_message, temperature, filter_key_values, search_score_cutoff, chat_config, attachments)

    def create_batch_prediction(self, table_name: str = None, name: str = None, global_prediction_args: Union[dict, BatchPredictionArgs] = None, batch_prediction_args: Union[dict, BatchPredictionArgs] = None, explanations: bool = False, output_format: str = None, output_location: str = None, database_connector_id: str = None, database_output_config: dict = None, refresh_schedule: str = None, csv_input_prefix: str = None, csv_prediction_prefix: str = None, csv_explanations_prefix: str = None, output_includes_metadata: bool = None, result_input_columns: list = None, input_feature_groups: dict = None):
        """
        Creates a batch prediction job description for the given deployment.

        Args:
            table_name (str): Name of the feature group table to write the results of the batch prediction. Can only be specified if outputLocation and databaseConnectorId are not specified. If tableName is specified, the outputType will be enforced as CSV.
            name (str): Name of the batch prediction job.
            batch_prediction_args (BatchPredictionArgs): Batch Prediction args specific to problem type.
            output_format (str): Format of the batch prediction output (CSV or JSON).
            output_location (str): Location to write the prediction results. Otherwise, results will be stored in Abacus.AI.
            database_connector_id (str): Unique identifier of a Database Connection to write predictions to. Cannot be specified in conjunction with outputLocation.
            database_output_config (dict): Key-value pair of columns/values to write to the database connector. Only available if databaseConnectorId is specified.
            refresh_schedule (str): Cron-style string that describes a schedule in UTC to automatically run the batch prediction.
            csv_input_prefix (str): Prefix to prepend to the input columns, only applies when output format is CSV.
            csv_prediction_prefix (str): Prefix to prepend to the prediction columns, only applies when output format is CSV.
            csv_explanations_prefix (str): Prefix to prepend to the explanation columns, only applies when output format is CSV.
            output_includes_metadata (bool): If true, output will contain columns including prediction start time, batch prediction version, and model version.
            result_input_columns (list): If present, will limit result files or feature groups to only include columns present in this list.
            input_feature_groups (dict): A dict of {'<feature_group_type>': '<feature_group_id>'} which overrides the default input data of that type for the Batch Prediction. Default input data is the training data that was used for training the deployed model.

        Returns:
            BatchPrediction: The batch prediction description.
        """
        return self.client.create_batch_prediction(self.deployment_id, table_name, name, global_prediction_args, batch_prediction_args, explanations, output_format, output_location, database_connector_id, database_output_config, refresh_schedule, csv_input_prefix, csv_prediction_prefix, csv_explanations_prefix, output_includes_metadata, result_input_columns, input_feature_groups)

    def get_statistics_over_time(self, start_date: str, end_date: str):
        """
        Return basic access statistics for the given window

        Args:
            start_date (str): Timeline start date in ISO format.
            end_date (str): Timeline end date in ISO format. The date range must be 7 days or less.

        Returns:
            DeploymentStatistics: Object describing Time series data of the number of requests and latency over the specified time period.
        """
        return self.client.get_deployment_statistics_over_time(self.deployment_id, start_date, end_date)

    def describe_feature_group_row_process_by_key(self, primary_key_value: str):
        """
        Gets the feature group row process.

        Args:
            primary_key_value (str): The primary key value

        Returns:
            FeatureGroupRowProcess: An object representing the feature group row process
        """
        return self.client.describe_feature_group_row_process_by_key(self.deployment_id, primary_key_value)

    def list_feature_group_row_processes(self, limit: int = None, status: str = None):
        """
        Gets a list of feature group row processes.

        Args:
            limit (int): The maximum number of processes to return. Defaults to None.
            status (str): The status of the processes to return. Defaults to None.

        Returns:
            list[FeatureGroupRowProcess]: A list of object representing the feature group row process
        """
        return self.client.list_feature_group_row_processes(self.deployment_id, limit, status)

    def get_feature_group_row_process_summary(self):
        """
        Gets a summary of the statuses of the individual feature group processes.

        Args:
            deployment_id (str): The deployment id for the process

        Returns:
            FeatureGroupRowProcessSummary: An object representing the summary of the statuses of the individual feature group processes
        """
        return self.client.get_feature_group_row_process_summary(self.deployment_id)

    def reset_feature_group_row_process_by_key(self, primary_key_value: str):
        """
        Resets a feature group row process so that it can be reprocessed

        Args:
            primary_key_value (str): The primary key value

        Returns:
            FeatureGroupRowProcess: An object representing the feature group row process.
        """
        return self.client.reset_feature_group_row_process_by_key(self.deployment_id, primary_key_value)

    def get_feature_group_row_process_logs_by_key(self, primary_key_value: str):
        """
        Gets the logs for a feature group row process

        Args:
            primary_key_value (str): The primary key value

        Returns:
            FeatureGroupRowProcessLogs: An object representing the logs for the feature group row process
        """
        return self.client.get_feature_group_row_process_logs_by_key(self.deployment_id, primary_key_value)

    def create_conversation(self, name: str = None, external_application_id: str = None):
        """
        Creates a deployment conversation.

        Args:
            name (str): The name of the conversation.
            external_application_id (str): The external application id associated with the deployment conversation.

        Returns:
            DeploymentConversation: The deployment conversation.
        """
        return self.client.create_deployment_conversation(self.deployment_id, name, external_application_id)

    def list_conversations(self, external_application_id: str = None, conversation_type: Union[dict, DeploymentConversationType] = None, fetch_last_llm_info: bool = False, limit: int = None, search: str = None):
        """
        Lists all conversations for the given deployment and current user.

        Args:
            external_application_id (str): The external application id associated with the deployment conversation. If specified, only conversations created on that application will be listed.
            conversation_type (DeploymentConversationType): The type of the conversation indicating its origin.
            fetch_last_llm_info (bool): If true, the LLM info for the most recent conversation will be fetched. Only applicable for system-created bots.
            limit (int): The number of conversations to return. Defaults to 600.
            search (str): The search query to filter conversations by title.

        Returns:
            list[DeploymentConversation]: The deployment conversations.
        """
        return self.client.list_deployment_conversations(self.deployment_id, external_application_id, conversation_type, fetch_last_llm_info, limit, search)

    def create_external_application(self, name: str = None, description: str = None, logo: str = None, theme: dict = None):
        """
        Creates a new External Application from an existing ChatLLM Deployment.

        Args:
            name (str): The name of the External Application. If not provided, the name of the deployment will be used.
            description (str): The description of the External Application. This will be shown to users when they access the External Application. If not provided, the description of the deployment will be used.
            logo (str): The logo to be displayed.
            theme (dict): The visual theme of the External Application.

        Returns:
            ExternalApplication: The newly created External Application.
        """
        return self.client.create_external_application(self.deployment_id, name, description, logo, theme)

    def download_agent_attachment(self, attachment_id: str):
        """
        Return an agent attachment.

        Args:
            attachment_id (str): The attachment ID.
        """
        return self.client.download_agent_attachment(self.deployment_id, attachment_id)

    def wait_for_deployment(self, wait_states={'PENDING', 'DEPLOYING'}, timeout=900):
        """
        A waiting call until deployment is completed.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, wait_states, timeout=timeout)

    def wait_for_pending_deployment_update(self, timeout=900):
        """
        A waiting call until deployment is in a stable state, that pending model switch is completed and previous model is stopped.

        Args:
            timeout (int): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.

        Returns:
            Deployment: the latest deployment object.
        """
        import time
        start_time = time.time()
        while True:
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f'Maximum wait time of {timeout}s exceeded')
            self.refresh()
            if not self.pending_model_version and not self.pending_prediction_operator_version:
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
