from .return_class import AbstractApiClass


class ModelMonitorVersion(AbstractApiClass):
    """
        A version of a model monitor

        Args:
            client (ApiClient): An authenticated API Client instance
            modelMonitorVersion (str): The unique identifier of a model monitor version.
            status (str): The current status of the model.
            modelMonitorId (str): A reference to the model monitor this version belongs to.
            monitoringStartedAt (str): The start time and date of the monitoring process.
            monitoringCompletedAt (str): The end time and date of the monitoring process.
            trainingFeatureGroupVersion (list[str]): Feature group version IDs that this refresh pipeline run is monitoring.
            predictionFeatureGroupVersion (list[str]): Feature group version IDs that this refresh pipeline run is monitoring.
            error (str): Relevant error if the status is FAILED.
            pendingDeploymentIds (list): List of deployment IDs where deployment is pending.
            failedDeploymentIds (list): List of failed deployment IDs.
            metricConfigs (list[dict]): List of metric configs for the model monitor instance.
            featureGroupMonitorConfigs (dict): Configurations for feature group monitor
            metricTypes (list): List of metric types.
            modelVersion (list[str]): Model version IDs that this refresh pipeline run is monitoring.
            batchPredictionVersion (str): The batch prediction version this model monitor is monitoring
            edaConfigs (list): The list of eda configs for the version
    """

    def __init__(self, client, modelMonitorVersion=None, status=None, modelMonitorId=None, monitoringStartedAt=None, monitoringCompletedAt=None, trainingFeatureGroupVersion=None, predictionFeatureGroupVersion=None, error=None, pendingDeploymentIds=None, failedDeploymentIds=None, metricConfigs=None, featureGroupMonitorConfigs=None, metricTypes=None, modelVersion=None, batchPredictionVersion=None, edaConfigs=None):
        super().__init__(client, modelMonitorVersion)
        self.model_monitor_version = modelMonitorVersion
        self.status = status
        self.model_monitor_id = modelMonitorId
        self.monitoring_started_at = monitoringStartedAt
        self.monitoring_completed_at = monitoringCompletedAt
        self.training_feature_group_version = trainingFeatureGroupVersion
        self.prediction_feature_group_version = predictionFeatureGroupVersion
        self.error = error
        self.pending_deployment_ids = pendingDeploymentIds
        self.failed_deployment_ids = failedDeploymentIds
        self.metric_configs = metricConfigs
        self.feature_group_monitor_configs = featureGroupMonitorConfigs
        self.metric_types = metricTypes
        self.model_version = modelVersion
        self.batch_prediction_version = batchPredictionVersion
        self.eda_configs = edaConfigs

    def __repr__(self):
        return f"ModelMonitorVersion(model_monitor_version={repr(self.model_monitor_version)},\n  status={repr(self.status)},\n  model_monitor_id={repr(self.model_monitor_id)},\n  monitoring_started_at={repr(self.monitoring_started_at)},\n  monitoring_completed_at={repr(self.monitoring_completed_at)},\n  training_feature_group_version={repr(self.training_feature_group_version)},\n  prediction_feature_group_version={repr(self.prediction_feature_group_version)},\n  error={repr(self.error)},\n  pending_deployment_ids={repr(self.pending_deployment_ids)},\n  failed_deployment_ids={repr(self.failed_deployment_ids)},\n  metric_configs={repr(self.metric_configs)},\n  feature_group_monitor_configs={repr(self.feature_group_monitor_configs)},\n  metric_types={repr(self.metric_types)},\n  model_version={repr(self.model_version)},\n  batch_prediction_version={repr(self.batch_prediction_version)},\n  eda_configs={repr(self.eda_configs)})"

    def to_dict(self):
        """
        Get a dict representation of the parameters in this class

        Returns:
            dict: The dict value representation of the class parameters
        """
        return {'model_monitor_version': self.model_monitor_version, 'status': self.status, 'model_monitor_id': self.model_monitor_id, 'monitoring_started_at': self.monitoring_started_at, 'monitoring_completed_at': self.monitoring_completed_at, 'training_feature_group_version': self.training_feature_group_version, 'prediction_feature_group_version': self.prediction_feature_group_version, 'error': self.error, 'pending_deployment_ids': self.pending_deployment_ids, 'failed_deployment_ids': self.failed_deployment_ids, 'metric_configs': self.metric_configs, 'feature_group_monitor_configs': self.feature_group_monitor_configs, 'metric_types': self.metric_types, 'model_version': self.model_version, 'batch_prediction_version': self.batch_prediction_version, 'eda_configs': self.eda_configs}

    def get_prediction_drift(self):
        """
        Gets the label and prediction drifts for a model monitor.

        Args:
            model_monitor_version (str): Unique string identifier for a model monitor version created under the project.

        Returns:
            DriftDistributions: Object describing training and prediction output label and prediction distributions.
        """
        return self.client.get_prediction_drift(self.model_monitor_version)

    def refresh(self):
        """
        Calls describe and refreshes the current object's fields

        Returns:
            ModelMonitorVersion: The current object
        """
        self.__dict__.update(self.describe().__dict__)
        return self

    def describe(self):
        """
        Retrieves a full description of the specified model monitor version.

        Args:
            model_monitor_version (str): The unique version ID of the model monitor version.

        Returns:
            ModelMonitorVersion: A model monitor version.
        """
        return self.client.describe_model_monitor_version(self.model_monitor_version)

    def delete(self):
        """
        Deletes the specified model monitor version.

        Args:
            model_monitor_version (str): Unique identifier of the model monitor version to delete.
        """
        return self.client.delete_model_monitor_version(self.model_monitor_version)

    def metric_data(self, metric_type: str, actual_values_to_detail: list = None):
        """
        Provides the data needed for decile metrics associated with the model monitor.

        Args:
            metric_type (str): The type of metric to get data for.
            actual_values_to_detail (list): The actual values to detail.

        Returns:
            ModelMonitorVersionMetricData: Data associated with the metric.
        """
        return self.client.model_monitor_version_metric_data(self.model_monitor_version, metric_type, actual_values_to_detail)

    def list_monitor_alert_versions_for_monitor_version(self):
        """
        Retrieves the list of monitor alert versions for a specified monitor instance.

        Args:
            model_monitor_version (str): The unique ID associated with the model monitor.

        Returns:
            list[MonitorAlertVersion]: A list of monitor alert versions.
        """
        return self.client.list_monitor_alert_versions_for_monitor_version(self.model_monitor_version)

    def get_model_monitoring_logs(self, stdout: bool = False, stderr: bool = False):
        """
        Returns monitoring logs for the model.

        Args:
            stdout (bool): Set True to get info logs
            stderr (bool): Set True to get error logs

        Returns:
            FunctionLogs: A function logs.
        """
        return self.client.get_model_monitoring_logs(self.model_monitor_version, stdout, stderr)

    def get_drift_for_feature(self, feature_name: str, nested_feature_name: str = None):
        """
        Gets the feature drift associated with a single feature in an output feature group from a prediction.

        Args:
            feature_name (str): Name of the feature to view the distribution of.
            nested_feature_name (str): Optionally, the name of the nested feature that the feature is in.

        Returns:
            FeatureDistribution: An object describing the training and prediction output feature distributions.
        """
        return self.client.get_drift_for_feature(self.model_monitor_version, feature_name, nested_feature_name)

    def get_outliers_for_feature(self, feature_name: str = None, nested_feature_name: str = None):
        """
        Gets a list of outliers measured by a single feature (or overall) in an output feature group from a prediction.

        Args:
            feature_name (str): Name of the feature to view the distribution of.
            nested_feature_name (str): Optionally, the name of the nested feature that the feature is in.
        """
        return self.client.get_outliers_for_feature(self.model_monitor_version, feature_name, nested_feature_name)

    def wait_for_monitor(self, timeout=1200):
        """
        A waiting call until model monitor version is ready.

        Args:
            timeout (int, optional): The waiting time given to the call to finish, if it doesn't finish by the allocated time, the call is said to be timed out.
        """
        return self.client._poll(self, {'PENDING', 'MONITORING'}, timeout=timeout)

    def get_status(self):
        """
        Gets the status of the model monitor version.

        Returns:
            str: A string describing the status of the model monitor version, for e.g., pending, complete, etc.
        """
        return self.describe().status
